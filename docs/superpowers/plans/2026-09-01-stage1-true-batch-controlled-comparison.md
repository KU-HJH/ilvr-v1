# Stage 1 True-Batch Controlled Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and execute the approved Stage 1 DoRA B=1/GA=4 versus true B=2/GA=2 controlled comparison at world size four/global batch 16, preserving every controllable sample, loss, stochastic, EMA, optimizer, scheduler, and launch semantic before comparing the two fresh 648-step runs and their official evaluations.

**Architecture:** A new `stage1.equivalence` module owns the canonical padded schedule, closed manifests, sample-keyed RNG, exact control records, numerical gates, and capacity-plan logic. Existing Stage 1 components remain the execution path: the collator processes each source row independently, the target builder runs frozen teacher work samplewise, and `CustomTrainerStage1` packs detached visual/teacher replacements into one autograd-safe true-batch decoder call. The existing `run_stage1_training.sh` and `python -m src.multinode train-stage1` route remain the only distributed training launcher; equivalence modes add bounded callbacks and evidence, not a second orchestration system.

**Tech Stack:** Python 3.11, PyTorch 2.7.1 + CUDA 12.8, local Transformers Qwen2.5-VL fork, PEFT 0.17.1, TRL/SFTTrainer, Accelerate, DeepSpeed ZeRO-1, FlashAttention 2.8.3, pytest, Bash, two-node four-GPU DDP.

**Spec:** `docs/superpowers/specs/2026-09-01-stage1-batch-equivalence-design.md`

## Global Constraints

- Use only `./venv/ilvr-venv` for Python, pytest, training, and evaluation.
- Before implementation, invoke `superpowers:using-git-worktrees` and create or verify an isolated worktree for the implementation branch. Do not overwrite the user's unrelated untracked files in `/root/workspace/ILVR`.
- Follow `superpowers:test-driven-development` for every behavior change. Confirm each new causal test is RED before changing production code, then run only its focused GREEN command.
- If a test, numerical gate, OOM, wrapper, DeepSpeed, FlashAttention, or distributed run fails unexpectedly, invoke `superpowers:systematic-debugging`, identify one root cause, add one causal regression test, and retry under a fresh run ID. Do not add generalized recovery infrastructure.
- The command blocks below name the first planned attempt. If a runtime defect is fixed, preserve that failed run and use the same role with suffix `-r2` (then the next unused integer only if a distinct later defect occurs). Exclusively publish `results/s1eq-v20-paired/run-selection.json` mapping each role to its one accepted run ID; every downstream reference resolves through that mapping. Selection never converts a numerical gate failure into a retry and never changes a threshold.
- Store every model-bearing tensor, adapter, optimizer/scheduler/RNG state, EMA snapshot, and comparison anchor only under `./checkpoints/<run-id>/`; `./checkpoints` must remain the existing symlink.
- Store JSON/JSONL metrics, comparisons, predictions, and summaries only under `./results/<run-id>/`; store all execution logs only under `./logs/<run-id>/` or the existing `./logs/multinode` control tree.
- Do not place model-bearing compatibility copies or symlinks under `results/`, and do not grow the repository filesystem with checkpoints.
- The only production model is `checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct`; the only training and evaluation inputs are `data/TRAIN.jsonl`, `data/images_comt`, and `data/TEST.jsonl` with the digests frozen by the spec.
- Fix the production experiment to DoRA `r=32`, `alpha=64`, dropout `0.05`, targets `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`, EMA tau `0.999`, LR `1e-5`, BF16, FlashAttention 2, non-reentrant gradient checkpointing, and `configs/deepspeed_zero1.json`.
- Fix world size to four, remote `korea@10.1.102.41` container `llm_kd` as rank-0/master, local as rank 1, two GPUs per node, `MASTER_ADDR=10.1.102.41`, and NCCL/GLOO interface `bond1.102`.
- The remote host checkout `/home/korea/Workspace/hjh/workspace/ILVR` is mounted as `/root/workspace/ILVR` inside `llm_kd`; all commands inside the container use only the `/root` path.
- Execute exactly one capacity smoke, two one-step replays, two 648-step mains, and two official evaluations in the order approved by the spec. Do not add full-horizon A/A, extra seeds, historical reproduction, or retry at relaxed thresholds.
- The fixed first-step gates are normative and immutable: same-arm `rtol=1e-6, atol=1e-7`; cross-arm symmetric relative L2 `<=0.02`, cosine `>=0.9998`; global CE/alignment/total absolute difference `<=0.01`; samplewise visual/teacher tolerance `rtol=1e-6, atol=1e-6`.
- The capacity smoke may use a validated preprocessing cache and may execute at most two optimizer updates. Main runs may not use that smoke-cache exemption.
- Keep Stage 2 behavior, generic generation, and DeepSpeed internals unchanged.
- Commit after each completed implementation task. Do not push, merge, publish a PR, or delete a worktree without explicit user approval.

## Normative Constants Carried from the Approved Spec

The implementation must not require a worker to infer these values from prose:

```text
TRAIN SHA-256     5bc1313a594f5713975b67388c745c6632275c37fd260055b27230c9cfedbed8
TEST SHA-256      6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618
ZeRO-1 SHA-256    c0922c8640778ae928c503f5bcacb070f2efaf34b7122431d9b4c2989e25debd
source rows       3,450
padded slots      3,456
updates/epoch     216
epochs/updates    3 / 648
dataloader workers 0
```

`cross_arm_v1` permits exactly:

```text
/batch/per_device_train_batch_size
/batch/gradient_accumulation_steps
/batch/rank_local_microbatch_grouping
/artifacts/run_id
/artifacts/checkpoints
/artifacts/results
/artifacts/logs
```

`same_arm_v1` permits exactly:

```text
/diagnostics/bounded_optimizer_steps
/artifacts/run_id
/artifacts/checkpoints
/artifacts/results
/artifacts/logs
```

The resolved manifest has exactly these top-level sections:
`schema_version`, `model`, `code`, `data`, `batch`, `schedule`, `tuning`,
`optimizer`, `scheduler`, `ema`, `precision`, `stochastic`, `deepspeed`,
`evaluation`, `diagnostics`, `artifacts`, and `initialization`. The exact child
schema is a versioned constant in `src/stage1/equivalence.py`; its tests build
one minimal complete object and apply delete/add/type mutations to every node.

Every `stage1-equivalence-trace-v1` optimizer-step record has exactly:
`schema_version`, `manifest_digest`, `normalized_manifest_digests`,
`route_fingerprint`, `arm_grouping_digest`, `rng_global_step`,
`optimizer_step_completed`, `epoch`, `update_in_epoch`, `counters`,
`control_digest`, `samples`, `observations`, and `runtime`. `counters` has
exactly `optimizer`, `scheduler`, `ema`, and `trainer_global_step`.

Each of the 16 sample records has exactly the occurrence/grouping integers;
`input_hashes` for `input_ids`, `labels`, `attention_mask`,
`original_pixel_values`, `helper_pixel_values`, `original_grid_thw`,
`helper_grid_thw`, and `position_ids`; sorted `stochastic_keys`; sorted
`dora_mask_specs`; `noise_bank_spec_digest`; hexadecimal-FP64 `sample_ce` and
`sample_alignment`; and ordered `teacher_blocks`. A teacher block contains
`latent_block_index`, `selected_count`, `selected_candidate_indices`, and a
hexadecimal-FP64 margin or `"all_selected"`. JSON numeric floats are forbidden.

For FP64 vectors `a,b`, symmetric relative L2 is
`2*||a-b||/(||a||+||b||)` and cosine is `dot(a,b)/(||a||*||b||)`. Two zero
vectors match exactly; a one-sided zero fails. The independently gated
first-step categories are final decoder hidden, accumulated pre-clip gradient,
student delta, EMA delta, Adam first moment, and Adam second moment. Global CE,
alignment, and total loss are reordered 16-sample FP64 means. These definitions
are duplicated here deliberately so the plan remains executable even before a
worker opens the linked design document.

---

## File Structure

### New production files

- `src/stage1/equivalence.py` — canonical schedule, closed manifests, canonical JSON/tensor hashes, sample RNG/context, route identity, trace schemas, comparison metrics/gates, capacity selection, and CLI inspection/comparison commands.
- `tests/stage1/test_equivalence.py` — schedule, manifest, RNG, schema, metrics, grouping, capacity, and report unit contracts.
- `tests/stage1/test_equivalence_integration.py` — two-step B=1/GA=4 versus B=2/GA=2 tiny training witness and independent EMA-order witness.

### Modified production files

- `src/stage1/types.py` — immutable occurrence, selection-diagnostic, stochastic-context, and equivalence-mode contracts.
- `src/stage1/data.py` — 3,450-to-3,456 scheduled view, occurrence propagation, independent per-row processor calls, row-local pixel/grid ownership, and deterministic packing.
- `src/stage1/targets.py` — samplewise teacher language/helper-visual execution, deterministic lower-index tie breaking, selected indices, and finite/tagged margins.
- `src/stage1/peft.py` — replacement of Stage 1 PEFT adapter dropout modules by registered sample-stable dropout sites.
- `src/stage1/trainer.py` — complete training-step stochastic context, samplewise frozen student vision, `torch.where` embedding composition, unified CE/alignment reductions, EMA/snapshot ordering, bounded stop, and online gates.
- `src/stage1/checkpointing.py` — schema-v3 stochastic/schedule resume identity and optimizer-boundary equivalence cursor sidecar.
- `src/stage1/composition.py` — canonical sampler/dataloader construction, 648-step horizon validation, manifest preflight, non-reentrant checkpoint context, diagnostic callback/evidence wiring, and capacity-only execution.
- `src/stage1/training_evidence.py` — equivalence trace/anchor publication and bounded/main completion evidence while retaining ordinary schema readers.
- `src/utils_deepseed.py` — closed equivalence CLI arguments and cross-field validation.
- `src/main.py` — direct-main artifact preflight for the added equivalence paths.
- `run_stage1_training.sh` — explicit equivalence arguments forwarded unchanged to the existing Accelerate launch.
- `src/multinode_runtime.py` — allowlist the new Stage 1 arguments and bind coordinator-owned equivalence evidence paths.
- `src/multinode.py` — recognize capacity/replay/main completion contracts without inventing checkpoints for the no-save capacity job.
- `src/multinode_artifacts.py` — collect only the closed per-rank equivalence-anchor paths into the remote rank-0 canonical run without treating them as inference artifacts.
- `docs/multinode-training.md` — exact staged commands, gate stop conditions, status paths, and official paired evaluation commands.

### Modified tests

- `tests/stage1/test_data.py`
- `tests/stage1/test_targets.py`
- `tests/stage1/test_peft.py`
- `tests/stage1/test_trainer.py`
- `tests/stage1/test_checkpointing.py`
- `tests/stage1/test_composition.py`
- `tests/stage1/test_training_evidence.py`
- `tests/stage1/test_launcher.py`
- `tests/multinode/test_coordinator.py`
- `tests/multinode/test_artifacts.py`
- `tests/multinode/test_topology.py`
- `tests/multinode/test_evaluation.py`

## Execution Dependency Map

Tasks 1–3 establish canonical identity and deterministic stochastic behavior. Tasks 4–6 replace the three currently grouping-sensitive computation paths: batched processor/vision, batched teacher, and branched/flat losses. Tasks 7–9 integrate Trainer lifecycle, EMA, resume, and evidence. Tasks 10–11 add manifest/capacity/multi-node launch plumbing without creating another launcher. Task 12 supplies the tiny framework witness. Task 13 is the focused review/verification gate. Tasks 14–17 are the only real 3B/GPU work and must run strictly in order; a failed gate stops subsequent expensive work. Task 18 evaluates and reports the two completed models without adding a new training variant.

### Task 1: Add the Canonical 3,456-Slot Schedule and Occurrence Identity

**Files:**
- Create: `src/stage1/equivalence.py`
- Create: `tests/stage1/test_equivalence.py`
- Modify: `src/stage1/types.py`
- Modify: `src/stage1/data.py`
- Modify: `tests/stage1/test_data.py`

**Interfaces:**
- Produces `Stage1SampleOccurrence(epoch, view_slot, permutation_offset, canonical_slot_id, source_index, rank, rank_local_sample_ordinal, rank_local_microbatch_ordinal, row_ordinal_within_microbatch)`.
- Produces `Stage1CanonicalDataset(source_rows, *, padded_length=3456)` and `Stage1CanonicalBatchSampler(dataset, *, world_size, seed, epochs, micro_batch_size, gradient_accumulation_steps, start_global_step=0)`.
- Produces `expected_stage1_assignment(*, epoch, update_in_epoch, rank, micro_batch_size) -> tuple[Stage1SampleOccurrence, ...]` and `validate_stage1_schedule(...) -> Stage1ScheduleSummary`.
- Extends `PreparedStage1Example` with `occurrence: Stage1SampleOccurrence | None`; ordinary Stage 1 permits `None`, equivalence mode requires a complete occurrence.

- [ ] **Step 1: Write failing padded-view and exact-assignment tests**

Add tests that assert exact source mapping and the approved arm grouping:

```python
def test_canonical_view_pads_3450_rows_to_3456_without_losing_provenance(rows_3450):
    view = Stage1CanonicalDataset(rows_3450)
    assert len(view) == 3456
    assert [view.source_index_for_slot(i) for i in range(3448, 3456)] == [
        3448, 3449, 0, 1, 2, 3, 4, 5,
    ]


@pytest.mark.parametrize(("micro_batch", "ga"), ((1, 4), (2, 2)))
def test_schedule_has_exact_216_updates_and_closed_rank_grouping(rows_3450, micro_batch, ga):
    schedule = materialize_stage1_schedule(
        rows_3450, rank=2, world_size=4, seed=42, epochs=3,
        micro_batch_size=micro_batch, gradient_accumulation_steps=ga,
    )
    assert schedule.updates_per_epoch == 216
    assert schedule.total_updates == 648
    first = schedule.update(0)
    assert [item.permutation_offset for item in first.samples] == [2, 6, 10, 14]
    expected_grouping = (
        [(0, 0), (1, 0), (2, 0), (3, 0)]
        if micro_batch == 1 else [(0, 0), (0, 1), (1, 0), (1, 1)]
    )
    assert [(x.rank_local_microbatch_ordinal, x.row_ordinal_within_microbatch) for x in first.samples] == expected_grouping
```

- [ ] **Step 2: Run the new schedule tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_equivalence.py::test_canonical_view_pads_3450_rows_to_3456_without_losing_provenance \
  tests/stage1/test_equivalence.py::test_schedule_has_exact_216_updates_and_closed_rank_grouping -q
```

Expected: import/attribute failures because the canonical schedule types do not exist.

- [ ] **Step 3: Implement immutable occurrence and schedule types**

Use a frozen occurrence whose invariants reject a slot/rank/grouping mismatch:

```python
@dataclass(frozen=True)
class Stage1SampleOccurrence:
    epoch: int
    view_slot: int
    permutation_offset: int
    canonical_slot_id: int
    source_index: int
    rank: int
    rank_local_sample_ordinal: int
    rank_local_microbatch_ordinal: int
    row_ordinal_within_microbatch: int
```

Implement the epoch permutation only through:

```python
generator = torch.Generator(device="cpu")
generator.manual_seed(42 + epoch)
permutation = torch.randperm(3456, generator=generator).tolist()
```

For global offset `16*u + 4*j + r`, assign B=1 to `(j, 0)` and B=2 to `(j // 2, j % 2)`. Set `canonical_slot_id = epoch * 3456 + view_slot`; never derive it from `source_index`.

- [ ] **Step 4: Make the dataset and collator propagate occurrence identity**

`Stage1CanonicalDataset.__getitem__` accepts the batch sampler's immutable scheduled index and returns `dataclasses.replace(source_row, occurrence=index.occurrence)`. Extend `Stage1SampleMetadata` with the occurrence fields and validate that all rows are either ordinary (`occurrence is None`) or equivalence rows (all complete); mixed batches fail.

The canonical sampler is a **global batch sampler**, not a rank-pre-sharded
sampler. It emits `world_size` microbatches for each rank-local microbatch
ordinal, ordered by rank. Accelerate's existing `BatchSamplerShard` with
`split_batches=False` then selects batch indices `r, r+4, ...` exactly once.
For B1 the global batches are singleton offsets `[4*j+r]`; for B2 they are
pairs `[4*(2*m)+r, 4*(2*m+1)+r]`. Set `batch_size` on the custom batch sampler,
make its length exactly divisible by four, set `even_batches=False`, and assert
after `accelerator.prepare` that the observed rank-local length/grouping is the
Section 4 mapping. Never combine it with `DistributedSampler` or another
rank-local sampler.

- [ ] **Step 5: Run focused schedule/data tests and confirm GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_equivalence.py \
  tests/stage1/test_data.py -q
```

Expected: 3,456-slot identity, B1/B2 grouping, duplicate-slot rejection, and existing source-ownership tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/stage1/equivalence.py src/stage1/types.py src/stage1/data.py \
  tests/stage1/test_equivalence.py tests/stage1/test_data.py
git commit -m "feat: add canonical Stage 1 comparison schedule"
```

### Task 2: Implement Canonical JSON, Tensor Hashes, and Closed Resolved Manifests

**Files:**
- Modify: `src/stage1/equivalence.py`
- Modify: `tests/stage1/test_equivalence.py`

**Interfaces:**
- Produces `canonical_json_bytes(value) -> bytes`, `canonical_tensor_sha256(tensor) -> str`, and `runtime_source_sha256(paths) -> str`.
- Produces `Stage1ResolvedManifest.from_mapping`, `.to_mapping`, `.digest`, and `.normalized_digest(profile)`.
- Produces `compare_stage1_manifests(left, right, *, profile: Literal["cross_arm_v1", "same_arm_v1"]) -> Stage1ManifestComparison`.
- Uses exact pointer sets from spec Section 2.1 and rejects duplicate JSON keys, unknown/missing keys, absent-vs-null substitution, arbitrary grouping, dirty tracked runtime source, nonfinite values, and unknown profiles.

- [ ] **Step 1: Write failing closed-schema and whitelist tests**

Include one parametrized mutation for every permitted pointer and representative forbidden fields:

```python
@pytest.mark.parametrize("pointer", CROSS_ARM_V1_POINTERS)
def test_cross_arm_manifest_accepts_only_the_seven_approved_pointers(manifest_pair, pointer):
    left, right = manifest_pair.changed_only_at(pointer)
    assert compare_stage1_manifests(left, right, profile="cross_arm_v1").equal


@pytest.mark.parametrize("pointer", (
    "/optimizer/learning_rate", "/ema/tau", "/stochastic/rng_schema",
    "/data/train_sha256", "/code/runtime_source_digest",
    "/deepspeed/settings/zero_optimization/stage",
))
def test_cross_arm_manifest_rejects_semantic_drift(manifest_pair, pointer):
    left, right = manifest_pair.changed_only_at(pointer)
    with pytest.raises(ValueError, match="non-whitelisted"):
        compare_stage1_manifests(left, right, profile="cross_arm_v1")
```

Also test duplicate JSON keys, missing child keys, extra child keys, invalid B1/B2 grouping, and canonical tensor hashes for dtype/shape/content changes.

- [ ] **Step 2: Run manifest tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_equivalence.py -k 'manifest or canonical_json or tensor_hash' -q
```

Expected: missing manifest/hash implementation failures.

- [ ] **Step 3: Implement the exact schema registry and pointer normalization**

Represent required keys as immutable nested schemas and compare after validating both unmodified objects:

```python
MANIFEST_PROFILES = {
    "cross_arm_v1": frozenset(CROSS_ARM_V1_POINTERS),
    "same_arm_v1": frozenset(SAME_ARM_V1_POINTERS),
}


def normalized_manifest_bytes(payload: Mapping[str, object], profile: str) -> bytes:
    validated = validate_resolved_manifest(payload)
    normalized = copy.deepcopy(validated)
    for pointer in MANIFEST_PROFILES[profile]:
        _replace_existing_json_pointer(normalized, pointer, "<approved-difference>")
    return canonical_json_bytes(normalized)
```

Use `json.loads(..., object_pairs_hook=_reject_duplicate_pairs)`, `json.dumps(sort_keys=True, separators=(",", ":"), allow_nan=False)`, and lowercase SHA-256. Tensor hashing must reject sparse/non-strided/nonfinite values, normalize integer byte order, and hash canonical `{dtype,shape}` bytes followed by contiguous little-endian CPU bytes.

- [ ] **Step 4: Implement manifest materialization from resolved runtime inputs**

Build every top-level section listed by the spec. Record paths as repository-relative logical identities and keep host/GPU/timing fields out of the manifest. Compute initialization identities in lexicographically sorted fully-qualified parameter-name order as `{name,dtype,shape,sha256}`. Resolve all DeepSpeed `auto` fields before writing; place microbatch/GA/effective batch only under `/batch`.

- [ ] **Step 5: Run manifest/hash tests and confirm GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_equivalence.py -k 'manifest or canonical_json or tensor_hash' -q
```

- [ ] **Step 6: Commit Task 2**

```bash
git add src/stage1/equivalence.py tests/stage1/test_equivalence.py
git commit -m "feat: seal Stage 1 comparison manifests"
```

### Task 3: Replace PEFT Adapter Dropout with Sample-Keyed Dropout

**Files:**
- Modify: `src/stage1/equivalence.py`
- Modify: `src/stage1/peft.py`
- Modify: `src/stage1/composition.py`
- Modify: `tests/stage1/test_equivalence.py`
- Modify: `tests/stage1/test_peft.py`
- Modify: `tests/stage1/test_composition.py`

**Interfaces:**
- Produces `stage1_sample_seed(base_seed, optimizer_global_step, canonical_slot_id, stable_site_name) -> int` using schema `stage1-sample-rng-v1`.
- Produces `Stage1StochasticContext`, `activate_stage1_stochastic_context(...)`, `stage1_checkpoint_context_fn()`, and `SampleStableStage1Dropout(site_name, p)`.
- Produces `install_stage1_sample_stable_dropout(model) -> tuple[str, ...]` and `validate_stage1_stochastic_inventory(model, expected_sites)`.

- [ ] **Step 1: Write failing grouping/padding/recompute tests**

Construct the same two unequal-length samples separately and packed, then require exact valid-prefix masks:

```python
def test_sample_dropout_is_grouping_and_right_padding_invariant(cuda_device):
    separate = run_dropout_rows(batch_sizes=(1, 1), lengths=(5, 8), device=cuda_device)
    packed = run_dropout_rows(batch_sizes=(2,), lengths=(5, 8), device=cuda_device)
    torch.testing.assert_close(packed[0][:5], separate[0][:5], rtol=0, atol=0)
    torch.testing.assert_close(packed[1][:8], separate[1][:8], rtol=0, atol=0)
    assert torch.equal(packed[0][5:], torch.ones_like(packed[0][5:]))


def test_checkpoint_recompute_regenerates_the_same_mask_and_rejects_second_call():
    original, recompute = run_original_and_recompute_once()
    torch.testing.assert_close(original, recompute, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="second.*recompute"):
        invoke_same_site_again_in_recompute()
```

Test missing context, wrong `[B,T,H]`, duplicate site names, unregistered active dropout, and a padded duplicate receiving a different seed from its original source row.

- [ ] **Step 2: Run focused dropout tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_equivalence.py -k 'sample_seed or dropout or recompute or stochastic_inventory' \
  tests/stage1/test_peft.py -q
```

- [ ] **Step 3: Implement context, seed derivation, and dropout**

Seed exactly as follows:

```python
payload = canonical_json_bytes({
    "base_seed": base_seed,
    "optimizer_global_step": optimizer_global_step,
    "canonical_slot_id": canonical_slot_id,
    "stable_site_name": stable_site_name,
})
seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)
```

The dropout validates `[B,T,H]`, context row count, each valid length, and site membership. Start from `torch.ones_like(x)`; for each row draw FP32 uniform values only for `[:valid_length, :]` with a fresh device-local generator, apply inverted dropout, and leave padding exactly one. Keep the PEFT `dropout(x)` call signature.

- [ ] **Step 4: Replace only resolved Stage 1 PEFT dropout sites**

After `get_peft_model`, walk `peft_model.base_model.targeted_module_names`, resolve each target's `lora_dropout["default"]`, replace it with `SampleStableStage1Dropout(fully_qualified_path, p)`, and return the sorted unique site names. Do not replace visual, base-model, or Stage 2 modules.

- [ ] **Step 5: Install non-reentrant checkpoint context and startup assertions**

Change Stage 1 DoRA/LoRA checkpointing to:

```python
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={
        "use_reentrant": False,
        "context_fn": stage1_checkpoint_context_fn,
    }
)
```

Do not rely only on `SFTConfig.gradient_checkpointing_kwargs`; after model preparation, inspect the resolved model checkpoint function/context and fail if the exact pair is absent. Full tuning remains outside this comparison and keeps its existing supported behavior.

- [ ] **Step 6: Add the fixed stochastic-source inventory test**

AST-scan `src/stage1/**/*.py` for the random APIs listed in spec Section 6.3. Allow only `canonical_epoch_permutation`, `SampleStableStage1Dropout`, and `alignment_noise_bank`. Inspect the loaded custom Qwen source and active `nn.Dropout` modules; reject any enabled unregistered site.

Set `dataloader_num_workers=0`, force student frozen vision and EMA teacher to
`eval()` at every training-step entry after the parent model enters train mode,
and record/set the same `torch.backends.cudnn.deterministic=True`,
`torch.backends.cudnn.benchmark=False`, TF32, and float32-matmul-precision values
in both manifests. FlashAttention/collective kernel nondeterminism remains the
approved measured same-arm risk; do not claim deterministic algorithms or
silently switch kernels to satisfy the gate.

- [ ] **Step 7: Run focused tests and confirm GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_equivalence.py -k 'sample_seed or dropout or recompute or stochastic_inventory' \
  tests/stage1/test_peft.py tests/stage1/test_composition.py -q
```

- [ ] **Step 8: Commit Task 3**

```bash
git add src/stage1/equivalence.py src/stage1/peft.py src/stage1/composition.py \
  tests/stage1/test_equivalence.py tests/stage1/test_peft.py tests/stage1/test_composition.py
git commit -m "feat: make Stage 1 adapter dropout sample-stable"
```

### Task 4: Make Processor and Pixel Ownership Samplewise Before Packing

**Files:**
- Modify: `src/stage1/types.py`
- Modify: `src/stage1/data.py`
- Modify: `tests/stage1/test_data.py`

**Interfaces:**
- Extends `Stage1SampleMetadata` with `original_pixel_range`, `helper_pixel_range`, and the complete occurrence identity.
- Produces `_collate_processed_rows(rows) -> dict[str, object]`, right-padding token tensors while concatenating pixel/grid tensors in row order.
- Keeps ordinary `Stage1DataCollator.__call__` public behavior and prohibits silent filtering/truncation.

- [ ] **Step 1: Write failing independent-processor and pixel-range tests**

Use a processor spy that changes results when called with two rows. Assert three processor calls per sample (`full`, `original`, `helper`), never one two-row call, and exact half-open pixel/grid ranges. Add a B=1-vs-packed test requiring each row's unpadded IDs, masks, labels, pixels, and grids to match exactly.

- [ ] **Step 2: Run data tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_data.py -k 'samplewise or pixel_range or packed_equals' -q
```

Expected: current collator sends the whole tuple to each processor call and has no pixel-row ranges.

- [ ] **Step 3: Process each `PreparedStage1Example` independently**

Inside `__call__`, call the current full/original/helper helpers with a singleton tuple. Normalize the latent block and build labels before any cross-row packing. Concatenate pixel rows and image-grid rows in source-row order; right-pad only `input_ids`, `attention_mask`, `labels`, and `image_out_mask`. Derive pixel ranges from the actual singleton tensor row counts, not from another row's grid or a global equal-split assumption.

- [ ] **Step 4: Validate exact ownership and no hidden truncation**

For each row, require `attention_mask.sum() == unpadded_length`, exact image-token/feature ownership, and all latent spans within the unpadded prefix. Reject duplicate occurrence IDs within an equivalence microbatch and reject any processor output that omits or shortens a source row.

- [ ] **Step 5: Run all Stage 1 data tests and confirm GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_data.py -q
```

- [ ] **Step 6: Commit Task 4**

```bash
git add src/stage1/types.py src/stage1/data.py tests/stage1/test_data.py
git commit -m "fix: preserve samplewise Stage 1 preprocessing"
```

### Task 5: Make EMA-Teacher Target Construction Samplewise and Deterministic

**Files:**
- Modify: `src/stage1/types.py`
- Modify: `src/stage1/targets.py`
- Modify: `tests/stage1/test_targets.py`

**Interfaces:**
- Extends `Stage1TeacherTargets` with `selected_candidate_indices` and `selection_margins` in sample/block order.
- Produces `Stage1TeacherTargetBuilder._build_sample(...)` which performs one unpadded language forward and one helper-visual forward per sample.
- Keeps the packed `[N,H]` values, `[B,T]` bool mask, and selected-count contract.

- [ ] **Step 1: Reverse the current batched-call assertion into a failing samplewise test**

For a two-row/two-plus-block case assert:

```python
assert case.teacher.language_forward_count == 2
assert case.teacher.helper_visual_forward_count == 2
assert case.teacher.language_batch_sizes == [1, 1]
assert case.teacher.helper_batches_are_row_owned
```

Require packed targets, counts, indices, and margins to equal concatenated independent B=1 calls. Add equal-similarity candidates and assert lower row-local candidate index wins.

- [ ] **Step 2: Run target tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_targets.py -k 'samplewise or independent or tie or margin' -q
```

Expected: current builder performs one batched language forward and one batched helper-visual call.

- [ ] **Step 3: Implement row extraction and one-call-per-role target construction**

For each metadata row, slice `input_ids[:unpadded_length]`, attention, row-owned original/helper pixels, and grids; call the teacher with batch dimension one and no padding from another sample. Reuse the single language/helper outputs across every latent block. Pack returned selections into the existing row-major mask/value order only after the row finishes.

- [ ] **Step 4: Implement deterministic selection diagnostics**

Use stable descending ordering with original candidate indices as the tie key. Return selected row-local indices in selection order. If at least one candidate remains excluded, record `last_selected_probability - first_excluded_probability` as a finite FP64 value; otherwise record the tagged string `"all_selected"`. Diagnostics never enter student inputs.

- [ ] **Step 5: Run all target tests and confirm GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_targets.py -q
```

- [ ] **Step 6: Commit Task 5**

```bash
git add src/stage1/types.py src/stage1/targets.py tests/stage1/test_targets.py
git commit -m "fix: build Stage 1 teacher targets samplewise"
```

### Task 6: Pack Samplewise Student Vision into One Autograd-Safe Decoder Call

**Files:**
- Modify: `src/stage1/trainer.py`
- Modify: `tests/stage1/test_trainer.py`
- Modify: `tests/stage1/test_qwen_latent_injection.py`

**Interfaces:**
- Produces `_build_stage1_student_inputs(model, working_inputs, metadata, targets) -> Stage1PackedStudentInputs`.
- Calls frozen student visual exactly once per row under `torch.no_grad()` and calls the composed PEFT student exactly once for the packed B=1/B=2 decoder tensor.
- Uses `input_ids`, prebuilt `inputs_embeds`, attention mask, explicit position IDs, `pixel_values=None`, and `latent_hidden_states=None`.

- [ ] **Step 1: Write a failing causal spy test**

Assert exact event order:

```python
assert events == [
    "student_visual_row_0",
    "student_visual_row_1",
    "student_decoder_batch_2",
]
assert student.forward_kwargs["pixel_values"] is None
assert student.forward_kwargs["latent_hidden_states"] is None
assert student.forward_kwargs["inputs_embeds"] is packed_inputs
```

Add a gradient test proving overwritten visual/teacher positions have zero input-embedding-row gradient, non-overwritten text/latent-token positions retain gradient, and decoder parameters receive finite gradients.

- [ ] **Step 2: Run student composition tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_trainer.py -k 'student_visual or autograd_safe or one_student_forward' \
  tests/stage1/test_qwen_latent_injection.py -q
```

- [ ] **Step 3: Build row-local frozen visual replacements**

Resolve the underlying Qwen module, then for each sample call `qwen.visual(row_pixels.to(qwen.visual.dtype), grid_thw=row_grid)` under `torch.no_grad()`. Verify the feature count equals the row's image-token count and write it to the corresponding positions of detached replacement storage.

- [ ] **Step 4: Compose embeddings without in-place autograd mutation**

Use exactly:

```python
token_embeds = qwen.model.embed_tokens(input_ids)
replacement_embeds = torch.zeros_like(token_embeds).detach()
replacement_mask = torch.zeros_like(input_ids, dtype=torch.bool)
# write detached row-local visual outputs and clean EMA targets into replacement_embeds
inputs_embeds = torch.where(
    replacement_mask.unsqueeze(-1), replacement_embeds, token_embeds
)
```

Compute multimodal position IDs from the original row grids and packed attention mask before nulling pixel inputs. Call the composed PEFT model once. Add a wrapper regression test proving the local Qwen forward skips token embedding, visual execution, and latent scatter when `inputs_embeds` is supplied.

- [ ] **Step 5: Run focused trainer/Qwen tests and confirm GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_trainer.py -k 'student_visual or autograd_safe or one_student_forward or helper_images_never' \
  tests/stage1/test_qwen_latent_injection.py -q
```

- [ ] **Step 6: Commit Task 6**

```bash
git add src/stage1/trainer.py tests/stage1/test_trainer.py \
  tests/stage1/test_qwen_latent_injection.py
git commit -m "fix: pack samplewise Stage 1 visual inputs safely"
```

### Task 7: Unify Per-Sample CE and Alignment with Sample-Keyed Noise

**Files:**
- Modify: `src/stage1/equivalence.py`
- Modify: `src/stage1/trainer.py`
- Modify: `tests/stage1/test_equivalence.py`
- Modify: `tests/stage1/test_trainer.py`

**Interfaces:**
- Produces `stage1_per_sample_ce(logits, labels) -> tuple[Tensor, Tensor]` returning scalar batch mean and `[B]` row losses.
- Produces `alignment_noise_bank(metadata, hidden_size, stochastic_context, device) -> tuple[Tensor, tuple[Tensor,...]]`.
- Produces `stage1_per_sample_alignment(predicted, clean_targets, latent_mask, noise_rows) -> tuple[Tensor, Tensor]`.

- [ ] **Step 1: Add failing B=1/B=2 loss-and-gradient equivalence tests**

Retain the unequal-token CE fixture and add unequal selected-latent counts. Supply identical explicit stochastic noise and assert:

```python
torch.testing.assert_close(loss_b2, (loss_b1_a + loss_b1_b) / 2, rtol=1e-6, atol=1e-7)
torch.testing.assert_close(grad_b2, (grad_b1_a + grad_b1_b) / 2, rtol=1e-6, atol=1e-7)
```

Require B=1 and B=2 both omit labels from the model forward. Add a counterexample with one versus three selected latents that fails under the current flattened mean.

- [ ] **Step 2: Run loss tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_trainer.py -k 'equal_sample or unequal_latent or legacy_golden' -q
```

- [ ] **Step 3: Remove the native-model-CE branch**

Always remove `labels` before the student forward, compute every row's shifted CE in FP32 1,024-token chunks, sum chunks in ascending offset, divide by that row's valid-token count, sum row scalars in row order, and divide by `B`. Reject any zero-valid-token row. Remove `model_ce_loss` from `_equal_sample_ce_loss` rather than retaining a B=1 compatibility branch.

- [ ] **Step 4: Implement per-row alignment and full latent-position noise banks**

For each sample, derive the `alignment_noise_bank` seed from its occurrence and upcoming optimizer step. Draw FP32 standard normal over all latent destination positions in metadata block/position order before selection. Gather only selected destination rows, add `0.01 * noise` to detached FP32 targets used by cosine loss, calculate a mean within each sample, then a mean over samples. Clean targets remain the decoder replacement values. A row with no selected latent position is an error.

- [ ] **Step 5: Run CUDA grouping tests and all trainer tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_equivalence.py -k 'noise_bank' \
  tests/stage1/test_trainer.py -q
```

- [ ] **Step 6: Commit Task 7**

```bash
git add src/stage1/equivalence.py src/stage1/trainer.py \
  tests/stage1/test_equivalence.py tests/stage1/test_trainer.py
git commit -m "fix: equalize Stage 1 sample loss weighting"
```

### Task 8: Bind Stochastic Context, EMA Ordering, and Snapshot Boundaries to `training_step`

**Files:**
- Modify: `src/stage1/trainer.py`
- Modify: `src/stage1/training_evidence.py`
- Modify: `tests/stage1/test_trainer.py`
- Modify: `tests/stage1/test_training_evidence.py`

**Interfaces:**
- Overrides `CustomTrainerStage1.training_step` to install one parent stochastic/evidence context around normal forward and backward, then validate site counts and release all samplewise intermediates in `finally`.
- Adds `Stage1EquivalenceRecorder` hooks at before-forward, target-ready, decoder-hidden, pre-clip gradient, post-optimizer/pre-EMA, post-EMA, and completed-step boundaries.
- Keeps `_finish_optimizer_step` as the sole successful EMA transition and preserves tau `1.0` fast no-op behavior.

- [ ] **Step 1: Write failing complete-context and independent EMA-order tests**

Add a test that calls the real Trainer `training_step`, confirms context remains active through backward recomputation, then is absent afterward even on exception. Add the independent E1 witness:

```python
expected_e1 = e0.clone()
expected_e1.mul_(0.999).add_(s1, alpha=0.001)
torch.testing.assert_close(live_teacher_before_step2, expected_e1, rtol=0, atol=0)
assert not torch.equal(live_teacher_before_step2[sentinel], e0[sentinel])
assert target_builder.teacher_state_seen_at_step2 == expected_e1
assert trainer._successful_optimizer_steps == 1
```

- [ ] **Step 2: Run lifecycle tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_trainer.py -k 'stochastic_context or independent_ema or optimizer_step' -q
```

- [ ] **Step 3: Override `training_step` without duplicating Trainer arithmetic**

Derive ordered `(canonical_slot_id, unpadded_length)` rows from batch metadata and `rng_global_step = self.state.global_step`. Set one `ContextVar` token, call `super().training_step(...)`, validate original/recompute invocation maps, clear retained intermediates, and reset exactly that token in `finally`. Do not manually divide loss or call backward outside the parent Trainer path.

- [ ] **Step 4: Add exact snapshot hooks around existing optimizer/EMA transitions**

Capture initialization before the first forward. Use the already version-pinned
`_DeepSpeedStepAdapter.__call__` as the mutation boundary. When
`engine.is_gradient_accumulation_boundary()` is true, call
`_capture_zero1_pre_step(engine)` immediately before `original_step`: clone the
already-reduced replicated trainable gradients in sorted parameter-name order,
divide the clones by the validated `engine.optimizer.loss_scale`, and leave the
live scaled gradients untouched before DeepSpeed performs norm-1.0 clipping.

Immediately after `original_step` returns and `engine.was_step_applied()` is
true, but before `_finish_optimizer_step`, call
`_capture_zero1_post_step_pre_ema(engine)`. Reconstruct logical Adam moments by
mapping DeepSpeed 0.18.0's ZeRO-1 FP32 partition/state slices back through the
validated ordered bit16 parameter groups, all-gathering only those state shards,
then splitting the full flat groups by canonical parameter shape. Validate the
pinned optimizer attributes and total element counts before accepting a value;
an unsupported layout is a launch error. Capture student delta there, call the
existing `_finish_optimizer_step` exactly once, and capture EMA delta afterward.
Use Trainer `on_step_end` only to observe the scheduler/global-step transition
and serialize the completed trace. A skipped/overflowed optimizer step emits no
completed-step trace and performs no EMA update.

- [ ] **Step 5: Run trainer/evidence tests and confirm GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_trainer.py \
  tests/stage1/test_training_evidence.py -q
```

- [ ] **Step 6: Commit Task 8**

```bash
git add src/stage1/trainer.py src/stage1/training_evidence.py \
  tests/stage1/test_trainer.py tests/stage1/test_training_evidence.py
git commit -m "feat: record ordered Stage 1 optimizer and EMA transitions"
```

### Task 9: Persist and Validate the Equivalence Resume Cursor

**Files:**
- Modify: `src/stage1/checkpointing.py`
- Modify: `src/stage1/composition.py`
- Modify: `src/stage1/trainer.py`
- Modify: `tests/stage1/test_checkpointing.py`
- Modify: `tests/stage1/test_composition.py`
- Modify: `tests/stage1/test_integration.py`

**Interfaces:**
- Bumps PEFT metadata to schema v3 with `sample_rng_schema`, `base_seed`, `padded_dataset_length`, `padded_dataset_digest`, `effective_global_batch`, and `schedule_schema`.
- Adds `stage1_equivalence_state.json` only to equivalence checkpoints with completed optimizer step, next epoch/update cursor, manifest digest, normalized digests, trace prefix digest, and stochastic schema.
- Produces `restore_stage1_equivalence_cursor(checkpoint, expected_manifest) -> Stage1ResumeCursor`.

- [ ] **Step 1: Write failing schema-v3 and real optimizer-boundary resume tests**

Require complete v2 read migration only for ordinary legacy artifacts; equivalence resume must require v3 fields. Train a tiny job to a boundary, save, resume, and assert the next four rank-local occurrences and regenerated dropout/noise specifications equal uninterrupted execution.

- [ ] **Step 2: Run checkpoint/resume tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_checkpointing.py -k 'schema_v3 or equivalence or rng_schema' \
  tests/stage1/test_integration.py -k 'equivalence_resume' -q
```

- [ ] **Step 3: Implement strict schema-v3 metadata and sidecar decoding**

Use exact-field validation, duplicate-key rejection, finite/integer range checks, canonical checkpoint ownership, and exclusive publication. A changed base seed, sample RNG schema, padded dataset digest/length, effective global batch, schedule schema, manifest digest, or trace-prefix digest fails before model load or dataloader construction.

- [ ] **Step 4: Restore the sampler before Trainer begins skipping data**

Compute `next_epoch, next_update_in_epoch = divmod(global_step, 216)`. Restore
the global canonical batch sampler before Trainer/Accelerate prepares it, call
its `set_epoch(next_epoch)`, and set its first global batch index to
`next_update_in_epoch * world_size * gradient_accumulation_steps`. Accelerate
then applies exactly one `BatchSamplerShard`; assert no `DistributedSampler`,
second `BatchSamplerShard`, reshuffle, or even-batch recycling appears. Ensure
the first resumed `training_step` sees the exact expected occurrence tuple
before recreating stochastic tensors. Keep checkpoints only at optimizer
boundaries.

- [ ] **Step 5: Run checkpoint/composition/integration tests and confirm GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_checkpointing.py \
  tests/stage1/test_composition.py \
  tests/stage1/test_integration.py -q
```

- [ ] **Step 6: Commit Task 9**

```bash
git add src/stage1/checkpointing.py src/stage1/composition.py src/stage1/trainer.py \
  tests/stage1/test_checkpointing.py tests/stage1/test_composition.py \
  tests/stage1/test_integration.py
git commit -m "feat: restore Stage 1 comparison schedule exactly"
```

### Task 10: Implement Trace, Anchor, and Fixed-Gate Comparisons

**Files:**
- Modify: `src/stage1/equivalence.py`
- Modify: `src/stage1/training_evidence.py`
- Modify: `src/stage1/trainer.py`
- Modify: `tests/stage1/test_equivalence.py`
- Modify: `tests/stage1/test_training_evidence.py`

**Interfaces:**
- Produces strict `stage1-equivalence-trace-v1` records and `Stage1TraceWriter` with one canonical rank-zero JSONL record per completed optimizer step.
- Produces `symmetric_relative_l2`, `cosine_similarity_fp64`, `compare_same_arm_step1`, `compare_cross_arm_step1`, and `compare_post_step_anchor`.
- Adds CLI commands `python -m stage1.equivalence compare-step1`, `compare-traces`, and `build-report`.
- Stores `.pt` anchors under `checkpoints/<run-id>/equivalence/`; writes only derived metrics/discrete traces under `results/<run-id>/`.

- [ ] **Step 1: Write failing exact-schema, grouping, and metric-edge tests**

Test every required top-level/sample/block field, duplicate/missing ranks, 15/17 samples, duplicate offsets, wrong arm row grouping, JSON numeric floats, nonfinite `float.hex` values, zero/one-sided-zero vectors, and lexicographic parameter concatenation. Require these decisions:

```python
assert symmetric_relative_l2(torch.zeros(3), torch.zeros(3)) == 0.0
assert compare_vector(torch.zeros(3), torch.ones(3)).accepted is False
assert compare_cross_arm_step1(passing_fixture).accepted is True
assert compare_cross_arm_step1(rel_l2_just_over_002).accepted is False
assert compare_same_arm_step1(atol_just_over_1e_7).accepted is False
```

- [ ] **Step 2: Run evidence/comparison tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_equivalence.py -k 'trace or grouping or relative_l2 or gate or report' \
  tests/stage1/test_training_evidence.py -k 'equivalence' -q
```

- [ ] **Step 3: Implement rank gathering and canonical control records**

Each rank submits exactly four sample records and one runtime record. Rank zero requires ranks 0–3 and offsets 0–15 relative to that update, sorts by integer permutation offset, reconstructs the exact arm grouping, and hashes it. `control_digest` includes only the spec's control fields; it excludes the approved grouping coordinates, teacher outcomes, losses, timings, and memory.

- [ ] **Step 4: Implement exact anchor layouts and numerical gates**

At step 1 store visual rows, clean/noisy teacher rows, valid decoder hidden rows, pre-clip gradients, student/EMA deltas, and logical Adam moments. At steps 2, 64, and 648 omit full decoder hidden but retain parameter/teacher anchors. Canonical vectors concatenate sorted parameter names with shape metadata; decoder hidden concatenates global permutation order then token/hidden order. Compare in FP64 and record relative L2, cosine, max absolute error, and worst parameter without using max error as a gate.

- [ ] **Step 5: Implement live same-arm and cross-arm gate callbacks**

Replay uses a dedicated `TrainerCallback.on_step_end`. It first asserts
`state.global_step == 1` and that the completed trace is staged, then sets both
`control.should_save = True` and `control.should_training_stop = True`. The
parent Trainer's normal `_maybe_log_save_evaluate` must publish `checkpoint-1`;
the Stage 1 checkpoint hook publishes EMA/equivalence sidecars and the trace
before the coordinator accepts completion. A causal test spies on the order
`completed_trace -> on_step_end -> parent_checkpoint_1 -> stop` and rejects an
exit with no complete checkpoint. The configured `save_steps=216` and 648-step
scheduler horizon remain equal to the corresponding main.

B1 main step 1 compares to B1 replay before continuation. B2 main step 1
compares to B2 replay and B1 main before continuation. Any gate failure
broadcasts failure to all ranks and stops without threshold mutation. B2 steps
2–648 compare exact control digests online to the completed B1 trace;
numerical/selection differences are recorded, not gated.

- [ ] **Step 6: Run trace/evidence tests and confirm GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_equivalence.py \
  tests/stage1/test_training_evidence.py \
  tests/stage1/test_trainer.py -q
```

- [ ] **Step 7: Commit Task 10**

```bash
git add src/stage1/equivalence.py src/stage1/training_evidence.py src/stage1/trainer.py \
  tests/stage1/test_equivalence.py tests/stage1/test_training_evidence.py \
  tests/stage1/test_trainer.py
git commit -m "feat: gate Stage 1 controlled comparison evidence"
```

### Task 11: Add Capacity Selection, Route Fingerprints, and Existing-Launcher Plumbing

**Files:**
- Modify: `src/stage1/equivalence.py`
- Modify: `src/stage1/composition.py`
- Modify: `src/stage1/trainer.py`
- Modify: `src/utils_deepseed.py`
- Modify: `src/main.py`
- Modify: `run_stage1_training.sh`
- Modify: `src/multinode_runtime.py`
- Modify: `src/multinode.py`
- Modify: `src/multinode_artifacts.py`
- Modify: `tests/stage1/test_equivalence.py`
- Modify: `tests/stage1/test_composition.py`
- Modify: `tests/stage1/test_trainer.py`
- Modify: `tests/stage1/test_launcher.py`
- Modify: `tests/multinode/test_coordinator.py`
- Modify: `tests/multinode/test_artifacts.py`
- Modify: `tests/multinode/test_topology.py`

**Interfaces:**
- Adds `--equivalence-mode {capacity,replay,main}`, `--equivalence-manifest`, `--equivalence-reference-run` (repeatable), `--equivalence-route-reference-run`, `--equivalence-results-dir`, `--equivalence-capacity-cache`, and `--equivalence-arm {b1,b2}`.
- Produces `Stage1CapacityPlan` from four risk axes and schema `stage1-capacity-preflight-v1`.
- Produces one canonical `route_fingerprint` gathered from four ranks.
- Keeps `run_stage1_training.sh` and `python -m src.multinode train-stage1` as the launch route.

- [ ] **Step 1: Write failing capacity selection and route tests**

Use synthetic scheduled metadata where each risk axis has a different maximizing pair. Require earliest tuple tie-break, deduplication, rank-zero max-count selection, even round-up, cyclic padding in fixed axis order, and at most two updates. Test that a changed GPU UUID, CUDA/NCCL version, rank mapping, master address, or interface changes/rejects the route fingerprint.

- [ ] **Step 2: Write failing CLI/coordinator tests**

Assert B1 accepts only microbatch/GA `1/4`, B2 only `2/2`, all modes require DoRA/ZeRO-1/world4/seed42/epochs3, replay requires the accepted capacity run only as `--equivalence-route-reference-run`, B1 main requires that route reference plus one B1 replay reference, B2 main requires the route reference plus B2 replay and B1 main references, and capacity rejects any model save. Before compute, every non-capacity mode loads the selected capacity fingerprint and rejects a different current route. Dry-run must show `run_stage1_training.sh` inside remote `docker exec llm_kd` and exact rank topology.

- [ ] **Step 3: Run focused plumbing tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_equivalence.py -k 'capacity or route' \
  tests/stage1/test_composition.py -k 'equivalence' \
  tests/stage1/test_trainer.py -k 'capacity' \
  tests/stage1/test_launcher.py -k 'equivalence' \
  tests/multinode/test_coordinator.py -k 'equivalence' \
  tests/multinode/test_artifacts.py -k 'equivalence' \
  tests/multinode/test_topology.py -k 'stage1' -q
```

- [ ] **Step 4: Implement four-axis capacity planning and cache validation**

For each scheduled B2 pair compute padded decoder length, original student visual elements, maximum per-row helper visual elements, and latent/candidate rows. Choose one pair per axis/rank with earliest `(epoch, update, microbatch)` tie-break, deduplicate, broadcast the largest selected count, round it to even, and cyclic-pad each rank. The resulting count is two or four pairs per rank. Preserve each selected pair's two original occurrence IDs; a cyclic repeat is allowed only in a later microbatch, never twice inside one pair. Group consecutive capacity pairs two-at-a-time under B2/GA2, use capacity-local RNG steps zero and one, and bypass the ordinary 648-step-length assertion only after validating this closed `Stage1CapacityPlan`. Execute exactly the resulting one or two optimizer updates. Validate cache schema plus TRAIN, processor, referenced-image, and cache payload digests before reuse; only capacity mode may read it.

- [ ] **Step 5: Implement the exact route object and fingerprint**

Gather sorted objects containing node role, machine/local/global rank, GPU UUID/model, driver, CUDA runtime, NCCL version, `MASTER_ADDR`, world size, and NCCL/GLOO interfaces. Reject any missing/extra field, duplicate rank/GPU, noncanonical role mapping, or value outside the approved route. Persist only canonical JSON and its SHA-256 in result evidence.

- [ ] **Step 6: Wire modes through existing launcher/coordinator**

Forward the new arguments as arrays through the shell launcher and allowlist.
`--equivalence-manifest` is an **exclusive output destination**: global rank zero
materializes it only after all ranks validate the same resolved semantic object,
then broadcasts its digest; it is never a preexisting input. Reference runs are
read from their already-published manifests.

Keep artifact ownership explicit. After a replay/main job, the existing
checkpoint collector first pushes local rank-2/3 checkpoint shards to the
remote rank-0 canonical checkpoint. A separate narrow anchor collector then
accepts only
`equivalence/{initialization,step-1,step-2,step-64,step-648}/rank-{0,1,2,3}.pt`
below `checkpoints/<run-id>/`, validates rank ownership and no symlinks, and
pushes the local rank-2/3 files into the remote canonical run. It rejects every
other path and never includes anchors in the inference manifest. Local storage
retains only its original rank shards. The coordinator pulls only small manifest/trace/completion JSON
into the matching local `results/<run-id>/`; it never pulls anchor tensors.
Live gates run on remote global rank zero against the remote canonical reference
anchors and broadcast accept/reject. Add the fixed coordinator command
`python -m src.multinode compare-stage1-equivalence` for offline comparisons:
it runs the allowlisted `stage1.equivalence` comparator inside `llm_kd`, reads
canonical remote anchors, and atomically publishes only the small derived JSON
locally.

Capacity success requires finite loss, no OOM, peak reserved on every rank
`<= 0.95 * total_memory`, at most two updates, no checkpoint/model artifact,
and only results/logs. In capacity mode route the small DeepSpeed runtime
identity sidecar to its `results/` evidence rather than Trainer `output_dir`,
disable every save callback, assert the preclaimed checkpoint run directory is
empty on both nodes, and remove that empty directory with `rmdir` after the
workers exit. Replay/main success keeps normal checkpoint collection
and additionally validates the equivalence completion evidence.

- [ ] **Step 7: Run focused plumbing tests and confirm GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_equivalence.py \
  tests/stage1/test_composition.py \
  tests/stage1/test_trainer.py \
  tests/stage1/test_launcher.py \
  tests/multinode/test_coordinator.py \
  tests/multinode/test_artifacts.py \
  tests/multinode/test_topology.py -q
```

- [ ] **Step 8: Commit Task 11**

```bash
git add src/stage1/equivalence.py src/stage1/composition.py src/stage1/trainer.py \
  src/utils_deepseed.py \
  src/main.py run_stage1_training.sh src/multinode_runtime.py src/multinode.py \
  src/multinode_artifacts.py \
  tests/stage1/test_equivalence.py tests/stage1/test_composition.py \
  tests/stage1/test_trainer.py \
  tests/stage1/test_launcher.py tests/multinode/test_coordinator.py \
  tests/multinode/test_artifacts.py \
  tests/multinode/test_topology.py
git commit -m "feat: route Stage 1 comparison through four GPUs"
```

### Task 12: Add the Two-Step Tiny Framework and EMA Witness

**Files:**
- Create: `tests/stage1/test_equivalence_integration.py`
- Modify: `tests/stage1/helpers.py`
- Modify: `tests/stage1/test_distributed_smoke.py`

**Interfaces:**
- Builds equal tiny Qwen/DoRA states and executes two optimizer steps under actual B=1/GA=4 and B=2/GA=2 Trainer arithmetic.
- Compares exact identities/counters and floating CE, alignment, accumulated gradient, Adam state, scheduler state, student delta, EMA delta at both steps.
- Independently constructs `E1` and proves step-2 teacher target reads it.

- [ ] **Step 1: Write the failing two-step witness before adapting fixtures**

The test must use the production collator, sample-stable dropout, target builder, Trainer `training_step`, optimizer boundary, and EMA update path. Do not replace the full step with direct helper calls. Require exact integer/discrete equality and `torch.testing.assert_close(rtol=1e-6, atol=1e-7)` for every floating category at both steps.

- [ ] **Step 2: Run the tiny witness and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_equivalence_integration.py -q
```

Expected: fixture/witness integration gaps are reported; do not weaken assertions.

- [ ] **Step 3: Add only the fixture support required by the production path**

Expose deterministic sample-owned visual rows, a tiny DoRA-compatible `[B,T,H]` decoder, explicit optimizer/scheduler state capture, and Trainer accumulation boundaries. Keep the production code path authoritative. Add an optional two-rank smoke assertion that all ranks agree on schedule counters and stochastic specs without requiring model-sized anchors.

- [ ] **Step 4: Run tiny and distributed-smoke tests and confirm GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_equivalence_integration.py \
  tests/stage1/test_distributed_smoke.py -q
```

- [ ] **Step 5: Commit Task 12**

```bash
git add tests/stage1/test_equivalence_integration.py tests/stage1/helpers.py \
  tests/stage1/test_distributed_smoke.py
git commit -m "test: witness Stage 1 batch arithmetic and EMA order"
```

### Task 13: Complete Focused Review, Verification, and Operator Documentation

**Files:**
- Modify: `docs/multinode-training.md`
- Review: all files changed in Tasks 1–12

- [ ] **Step 1: Document the exact staged operator commands and stop rules**

Document the seven fixed run roles (capacity, B1/B2 replay, B1/B2 main, and B1/B2 evaluation), ports, detached status commands, result/log/checkpoint paths, capacity no-save rule, replay/main reference arguments, immutable gates, and evaluation commands. `s1eq-v20-paired` is a result namespace, not an eighth run. State that B1 main starts only after replay/replay acceptance, B2 main starts only after B1 completion, and no threshold relaxation or automatic retry is allowed.

- [ ] **Step 2: Run `superpowers:requesting-code-review` on the implementation**

Request review against design commitment v20 with special attention to runtime call counts, autograd replacement semantics, stochastic checkpoint replay, EMA ordering, exact schedule placement, artifact roots, and the fact that capacity mode saves no model. Apply only verified findings through one causal test per fix.

- [ ] **Step 3: Run the single focused pre-GPU verification suite**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_equivalence.py \
  tests/stage1/test_equivalence_integration.py \
  tests/stage1/test_data.py \
  tests/stage1/test_targets.py \
  tests/stage1/test_peft.py \
  tests/stage1/test_trainer.py \
  tests/stage1/test_checkpointing.py \
  tests/stage1/test_composition.py \
  tests/stage1/test_training_evidence.py \
  tests/stage1/test_integration.py \
  tests/stage1/test_distributed_smoke.py \
  tests/stage1/test_qwen_latent_injection.py \
  tests/stage1/test_launcher.py \
  tests/multinode/test_coordinator.py \
  tests/multinode/test_artifacts.py \
  tests/multinode/test_topology.py \
  tests/multinode/test_evaluation.py -q
```

Expected: all focused tests pass once. Do not immediately rerun the same broad set.

- [ ] **Step 4: Check artifact paths and repository size before GPU work**

```bash
test -L checkpoints
test "$(readlink -f checkpoints)" = /raid/workspace/ILVR/checkpoints
if find results -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.bin' -o -name '*.safetensors' \) -print -quit | grep -q .; then
  exit 1
fi
git status --short
du -sh --exclude=checkpoints .
```

Expected: checkpoint symlink resolves to `/raid/workspace/ILVR/checkpoints`, no model-bearing result file is printed, only intended tracked changes/unrelated pre-existing user files appear, and the repository tree has not acquired model data.

- [ ] **Step 5: Commit Task 13**

```bash
git add docs/multinode-training.md
git commit -m "docs: add Stage 1 controlled comparison runbook"
```

- [ ] **Step 6: Finish the implementation branch before canonical-path execution**

Invoke `superpowers:verification-before-completion`, then
`superpowers:finishing-a-development-branch`. The multi-node coordinator is
intentionally fixed to `/root/workspace/ILVR`, so Tasks 14–18 may start only
after the user selects an integration option that makes the verified
implementation commit the clean tracked `HEAD` at that canonical path. Do not
silently merge or rebase the implementation branch.

### Task 14: Deploy the Exact Implementation Commit and Run Dry-Run Preflight

**Files:**
- No source changes expected.
- Produces only a small Git bundle outside the repository and dry-run output under `logs/`.

- [ ] **Step 1: Record and verify the implementation commit locally**

```bash
implementation_commit=$(git rev-parse HEAD)
bundle_name="ilvr-stage1-equivalence-${implementation_commit}.bundle"
git status --porcelain --untracked-files=no
./venv/ilvr-venv/bin/python -m src.multinode train-stage1 \
  --run-id s1eq-v20-capacity-b2 \
  --attempt-id initial \
  --zero-stage 1 \
  --main-process-port 30140 \
  --dry-run \
  --method dora \
  --model /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
  --data-path /root/workspace/ILVR/data/TRAIN.jsonl \
  --source-data-path /root/workspace/ILVR/data/TRAIN.jsonl \
  --image-root /root/workspace/ILVR/data/images_comt \
  --micro-batch 2 --gradient-accumulation 2 --epochs 3 \
  --learning-rate 1e-5 --ema-tau 0.999 --seed 42 \
  --attention-implementation flash_attention_2 \
  --save-strategy no --no-final-save \
  --equivalence-mode capacity --equivalence-arm b2 \
  --equivalence-manifest /root/workspace/ILVR/results/s1eq-v20-capacity-b2/resolved-manifest.json \
  --equivalence-results-dir /root/workspace/ILVR/results/s1eq-v20-capacity-b2 \
  --equivalence-capacity-cache /root/workspace/ILVR/cache/stage1-equivalence-v1
```

Expected: tracked tree is clean and dry-run shows remote rank 0 in `llm_kd`, local rank 1, four processes, ZeRO-1, master `10.1.102.41`, and interface `bond1.102`, without creating a run directory.

- [ ] **Step 2: Update the already-provisioned remote checkout to the exact commit**

First verify the remote tracked tree is clean; stop rather than overwrite if it is not:

```bash
implementation_commit=$(git rev-parse HEAD)
bundle_name="ilvr-stage1-equivalence-${implementation_commit}.bundle"
ssh -o BatchMode=yes -o ConnectTimeout=8 korea@10.1.102.41 \
  "/usr/bin/docker exec --workdir /root/workspace/ILVR llm_kd git status --porcelain --untracked-files=no"
git bundle create "/tmp/$bundle_name" HEAD
rsync -a --ignore-existing "/tmp/$bundle_name" \
  "korea@10.1.102.41:/home/korea/Workspace/hjh/$bundle_name"
ssh -o BatchMode=yes -o ConnectTimeout=8 korea@10.1.102.41 \
  "/usr/bin/docker exec --workdir /root/workspace/ILVR llm_kd git fetch '/root/$bundle_name' HEAD"
ssh -o BatchMode=yes -o ConnectTimeout=8 korea@10.1.102.41 \
  "/usr/bin/docker exec --workdir /root/workspace/ILVR llm_kd git checkout --detach '$implementation_commit'"
```

This is one-time deployment, not launcher-time synchronization. Do not transfer source trees, venvs, datasets, model weights, or checkpoints.

- [ ] **Step 3: Defer the one mutable preflight to the capacity coordinator**

Do not start a separate probe here. Task 15's existing coordinator performs the
commit, clean-tree, two-GPU, input-path, port, container, interface, and
artifact-root preflight exactly once immediately before it launches the
capacity workers. A preflight failure stops without creating a second launch.

### Task 15: Execute the Capacity Smoke and One-Step Replay Gates

**Files:**
- Runtime outputs only under canonical `results/` and `logs/`; replay anchors/checkpoints under `checkpoints/`.

- [ ] **Step 1: Launch the exact B2 capacity smoke detached**

```bash
install -d -m 700 /root/workspace/ILVR/logs/s1eq-v20-capacity-b2
tmux new-session -d -s s1eq-v20-capacity-control -- /usr/bin/bash -lc '
cd /root/workspace/ILVR &&
PYTHONPATH=/root/workspace/ILVR:/root/workspace/ILVR/src \
./venv/ilvr-venv/bin/python -m src.multinode train-stage1 \
  --run-id s1eq-v20-capacity-b2 --attempt-id initial --zero-stage 1 \
  --main-process-port 30140 \
  --method dora \
  --model /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
  --data-path /root/workspace/ILVR/data/TRAIN.jsonl \
  --source-data-path /root/workspace/ILVR/data/TRAIN.jsonl \
  --image-root /root/workspace/ILVR/data/images_comt \
  --micro-batch 2 --gradient-accumulation 2 --epochs 3 \
  --learning-rate 1e-5 --ema-tau 0.999 --seed 42 \
  --attention-implementation flash_attention_2 \
  --save-strategy no --no-final-save \
  --equivalence-mode capacity --equivalence-arm b2 \
  --equivalence-manifest /root/workspace/ILVR/results/s1eq-v20-capacity-b2/resolved-manifest.json \
  --equivalence-results-dir /root/workspace/ILVR/results/s1eq-v20-capacity-b2 \
  --equivalence-capacity-cache /root/workspace/ILVR/cache/stage1-equivalence-v1 \
  > /root/workspace/ILVR/logs/s1eq-v20-capacity-b2/coordinator.log 2>&1
'
```

Report immediately that the detached job started, with its coordinator/worker logs. Wait for the one capacity evidence object. Require finite updates, no OOM, `peak_reserved/total <= 0.95` on all ranks, identical route fingerprint, at most two updates, and no surviving `checkpoints/s1eq-v20-capacity-b2` directory.

- [ ] **Step 2: Launch B1 replay only after capacity passes**

```bash
install -d -m 700 /root/workspace/ILVR/logs/s1eq-v20-replay-b1
tmux new-session -d -s s1eq-v20-replay-b1-control -- /usr/bin/bash -lc '
cd /root/workspace/ILVR &&
PYTHONPATH=/root/workspace/ILVR:/root/workspace/ILVR/src \
./venv/ilvr-venv/bin/python -m src.multinode train-stage1 \
  --run-id s1eq-v20-replay-b1 --attempt-id initial --zero-stage 1 \
  --main-process-port 30141 \
  --method dora \
  --model /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
  --data-path /root/workspace/ILVR/data/TRAIN.jsonl \
  --source-data-path /root/workspace/ILVR/data/TRAIN.jsonl \
  --image-root /root/workspace/ILVR/data/images_comt \
  --micro-batch 1 --gradient-accumulation 4 --epochs 3 \
  --learning-rate 1e-5 --ema-tau 0.999 --seed 42 \
  --attention-implementation flash_attention_2 \
  --save-strategy steps --save-steps 216 --no-final-save \
  --equivalence-mode replay --equivalence-arm b1 \
  --equivalence-route-reference-run s1eq-v20-capacity-b2 \
  --equivalence-manifest /root/workspace/ILVR/results/s1eq-v20-replay-b1/resolved-manifest.json \
  --equivalence-results-dir /root/workspace/ILVR/results/s1eq-v20-replay-b1 \
  > /root/workspace/ILVR/logs/s1eq-v20-replay-b1/coordinator.log 2>&1
'
```

The bounded callback must save `checkpoint-1` and stop after exactly one optimizer update while the scheduler remains configured for 648.

- [ ] **Step 3: Launch B2 replay only after B1 replay validates**

```bash
install -d -m 700 /root/workspace/ILVR/logs/s1eq-v20-replay-b2
tmux new-session -d -s s1eq-v20-replay-b2-control -- /usr/bin/bash -lc '
cd /root/workspace/ILVR &&
PYTHONPATH=/root/workspace/ILVR:/root/workspace/ILVR/src \
./venv/ilvr-venv/bin/python -m src.multinode train-stage1 \
  --run-id s1eq-v20-replay-b2 --attempt-id initial --zero-stage 1 \
  --main-process-port 30142 \
  --method dora \
  --model /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
  --data-path /root/workspace/ILVR/data/TRAIN.jsonl \
  --source-data-path /root/workspace/ILVR/data/TRAIN.jsonl \
  --image-root /root/workspace/ILVR/data/images_comt \
  --micro-batch 2 --gradient-accumulation 2 --epochs 3 \
  --learning-rate 1e-5 --ema-tau 0.999 --seed 42 \
  --attention-implementation flash_attention_2 \
  --save-strategy steps --save-steps 216 --no-final-save \
  --equivalence-mode replay --equivalence-arm b2 \
  --equivalence-route-reference-run s1eq-v20-capacity-b2 \
  --equivalence-manifest /root/workspace/ILVR/results/s1eq-v20-replay-b2/resolved-manifest.json \
  --equivalence-results-dir /root/workspace/ILVR/results/s1eq-v20-replay-b2 \
  > /root/workspace/ILVR/logs/s1eq-v20-replay-b2/coordinator.log 2>&1
'
```

Require the capacity/replay route fingerprint to match exactly.

- [ ] **Step 4: Run the replay/replay cross-arm gate once**

```bash
PYTHONPATH="$PWD:$PWD/src" ./venv/ilvr-venv/bin/python -m src.multinode compare-stage1-equivalence \
  --kind step1 \
  --profile cross_arm_v1 \
  --left-run s1eq-v20-replay-b1 \
  --right-run s1eq-v20-replay-b2 \
  --output results/s1eq-v20-paired/replay-cross-arm-step1.json
```

Require exact controls/selection, visual/teacher same-shape tolerance, all six tensor-category gates, and all three scalar gates. If any gate fails, stop before both main runs and report the exact failing category without widening thresholds.

### Task 16: Execute the Fresh B1 and B2 648-Step Main Runs

**Files:**
- Runtime outputs only under `checkpoints/s1eq-v20-main-{b1,b2}`, `results/s1eq-v20-main-{b1,b2}`, and `logs/s1eq-v20-main-{b1,b2}`.

- [ ] **Step 1: Launch B1 main detached**

```bash
install -d -m 700 /root/workspace/ILVR/logs/s1eq-v20-main-b1
tmux new-session -d -s s1eq-v20-main-b1-control -- /usr/bin/bash -lc '
cd /root/workspace/ILVR &&
PYTHONPATH=/root/workspace/ILVR:/root/workspace/ILVR/src \
./venv/ilvr-venv/bin/python -m src.multinode train-stage1 \
  --run-id s1eq-v20-main-b1 --attempt-id initial --zero-stage 1 \
  --main-process-port 30143 \
  --method dora \
  --model /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
  --data-path /root/workspace/ILVR/data/TRAIN.jsonl \
  --source-data-path /root/workspace/ILVR/data/TRAIN.jsonl \
  --image-root /root/workspace/ILVR/data/images_comt \
  --micro-batch 1 --gradient-accumulation 4 --epochs 3 \
  --learning-rate 1e-5 --ema-tau 0.999 --seed 42 \
  --attention-implementation flash_attention_2 \
  --save-strategy steps --save-steps 216 --final-save \
  --equivalence-mode main --equivalence-arm b1 \
  --equivalence-route-reference-run s1eq-v20-capacity-b2 \
  --equivalence-reference-run s1eq-v20-replay-b1 \
  --equivalence-manifest /root/workspace/ILVR/results/s1eq-v20-main-b1/resolved-manifest.json \
  --equivalence-results-dir /root/workspace/ILVR/results/s1eq-v20-main-b1 \
  > /root/workspace/ILVR/logs/s1eq-v20-main-b1/coordinator.log 2>&1
'
```

The process must compare its actual step 1 to the replay before continuing uninterrupted to step 648.

- [ ] **Step 2: Validate B1 completion once**

Require completed traces 1–648, checkpoints at 216/432/648, final model under the same run directory, exact control consistency, finite observed values, and same-arm step-1 acceptance. Record elapsed time and ETA from observed steps without launching any other experiment while B1 runs.

- [ ] **Step 3: Launch B2 main only after B1 completes**

```bash
install -d -m 700 /root/workspace/ILVR/logs/s1eq-v20-main-b2
tmux new-session -d -s s1eq-v20-main-b2-control -- /usr/bin/bash -lc '
cd /root/workspace/ILVR &&
PYTHONPATH=/root/workspace/ILVR:/root/workspace/ILVR/src \
./venv/ilvr-venv/bin/python -m src.multinode train-stage1 \
  --run-id s1eq-v20-main-b2 --attempt-id initial --zero-stage 1 \
  --main-process-port 30144 \
  --method dora \
  --model /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
  --data-path /root/workspace/ILVR/data/TRAIN.jsonl \
  --source-data-path /root/workspace/ILVR/data/TRAIN.jsonl \
  --image-root /root/workspace/ILVR/data/images_comt \
  --micro-batch 2 --gradient-accumulation 2 --epochs 3 \
  --learning-rate 1e-5 --ema-tau 0.999 --seed 42 \
  --attention-implementation flash_attention_2 \
  --save-strategy steps --save-steps 216 --final-save \
  --equivalence-mode main --equivalence-arm b2 \
  --equivalence-route-reference-run s1eq-v20-capacity-b2 \
  --equivalence-reference-run s1eq-v20-replay-b2 \
  --equivalence-reference-run s1eq-v20-main-b1 \
  --equivalence-manifest /root/workspace/ILVR/results/s1eq-v20-main-b2/resolved-manifest.json \
  --equivalence-results-dir /root/workspace/ILVR/results/s1eq-v20-main-b2 \
  > /root/workspace/ILVR/logs/s1eq-v20-main-b2/coordinator.log 2>&1
'
```

At actual step 1 require both B2 same-arm and direct B1-main/B2-main cross-arm gates before continuing.

- [ ] **Step 4: Validate B2 online controls and completion once**

Require steps 2–648 to match the completed B1 trace on exact control digests/counters while treating teacher selections and numerical trajectory as observations. Require checkpoints 216/432/648, final model, finite observations, and no artifact outside canonical roots. Do not rerun either main for A/A.

### Task 17: Build the Paired Trajectory Comparison

**Files:**
- Produces `results/s1eq-v20-paired/trajectory-comparison.json` and metadata-only tables.

- [ ] **Step 1: Compare compact traces and anchors exactly once**

```bash
PYTHONPATH="$PWD:$PWD/src" ./venv/ilvr-venv/bin/python -m src.multinode compare-stage1-equivalence \
  --kind trajectory \
  --left-run s1eq-v20-main-b1 \
  --right-run s1eq-v20-main-b2 \
  --anchors 2 64 648 \
  --output results/s1eq-v20-paired/trajectory-comparison.json
```

Require exact controls for all 648 steps. At anchors report relative L2/cosine without applying a post-step-1 gate. Join teacher rows by `(canonical_slot_id, latent_block_index, candidate_index)`, report matched/unmatched counts, and identify the first selected-index divergence plus its margins.

- [ ] **Step 2: Verify the report does not overclaim**

The comparison must label A/A values as one observed discrepancy per arm, never a noise floor; must not subtract A/A or attribute excess to batching; and must keep semantic control, common-state numerical acceptance, and trajectory/operational observations in separate objects.

### Task 18: Run Official Evaluation and Publish the Separated Final Report

**Files:**
- Produces prepared model copies only under `checkpoints/evaluation-staging/`.
- Produces evaluation shards/summaries under `results/s1eq-v20-eval-{b1,b2}`.
- Produces `results/s1eq-v20-paired/final-report.json` and a concise Markdown table under the same result directory.

- [ ] **Step 1: Prepare each completed Stage 1 DoRA checkpoint for inference**

For each arm run:

```bash
./venv/ilvr-venv/bin/python -m src.multinode prepare-inference \
  --stage stage1 --tuning-method dora --zero-stage 1 \
  --run-id s1eq-v20-main-b1 --main-process-port 30143 \
  --checkpoint-step 648 \
  --destination /root/workspace/ILVR/checkpoints/evaluation-staging/s1eq-v20-main-b1/prepared-checkpoint-648
```

Repeat only with `b2` and port `30144`. Validate each prepared artifact locally and inside remote `llm_kd` using `prepared-inference-show`; identities must match.

- [ ] **Step 2: Evaluate B1 and B2 with the official four-shard settings**

```bash
./run_multinode_evaluation.sh \
  --run-id s1eq-v20-eval-b1 \
  --model-dir /root/workspace/ILVR/checkpoints/evaluation-staging/s1eq-v20-main-b1/prepared-checkpoint-648 \
  --base-model-dir /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
  --test-data-path /root/workspace/ILVR/data/TEST.jsonl \
  --task-name zebra-cot \
  --output-dir /root/workspace/ILVR/results/s1eq-v20-eval-b1 \
  --max-new-tokens 4096 --temperature 0 --top-p 1
```

Repeat for B2. Run sequentially. Require four valid source-index shards, complete coverage of 400 TEST rows, and exact scorer provenance from `original_final_answer`.

- [ ] **Step 3: Build one final paired report**

```bash
PYTHONPATH="$PWD:$PWD/src" ./venv/ilvr-venv/bin/python -m stage1.equivalence build-report \
  --b1-run s1eq-v20-main-b1 \
  --b2-run s1eq-v20-main-b2 \
  --b1-eval results/s1eq-v20-eval-b1 \
  --b2-eval results/s1eq-v20-eval-b2 \
  --output results/s1eq-v20-paired/final-report.json
```

Publish exact-correct counts/400 and accuracy side by side, join every prediction by zero-based TEST `source_index`, and list agreement/disagreement rows. Label evaluation as retrospective repository-test evidence, not untouched generalization.

- [ ] **Step 4: Run final claim-specific verification once**

Invoke `superpowers:verification-before-completion`. Verify: both traces have exactly 648 records; exact-control status is green; first-step same-arm and cross-arm results use unchanged fixed thresholds; both checkpoints are valid under `checkpoints/`; both evaluation summaries cover 400 rows; `results/` contains no model-bearing files; and the repository size did not grow through model artifacts.

- [ ] **Step 5: Report the three outcomes separately**

Report only:

1. semantic-control agreement through step 648;
2. common-state numerical gate results with B1 A/A, B2 A/A, replay A/B, and main A/B side by side;
3. operational/quality observations: total time, per-step time, peak memory, first EMA-selection divergence, anchor metrics, and B1/B2 accuracy.

Do not collapse them into a single “B2 success” statement and do not claim a causal/generalizable endpoint effect.
