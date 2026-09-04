# Packed Stage 2B Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Admit only authenticated Stage 2B examples whose unchanged natural
packed plan fits the 7168-token capacity envelope, prove the retained boundary
fits DoRA, and finish the CE-only, DPO-only, and entropy-only training and
evaluation runs without OOM.

**Architecture:** Reuse the existing strict Stage 2 length manifest, negative
manifest, production collator, packed planner, one-shot capacity probe, and
launcher. A new strict packed-admission module binds the clean source to one
byte-preserving retained source and independently verifies every reconstructed
plan. Normal packed Stage 2B training requires the shared admission and verifier
hashes; capacity-only probes use a private in-memory bypass and cannot be
enabled through the public CLI.

**Tech Stack:** Python 3.11, PyTorch 2.7.1/CUDA, Qwen2.5-VL, PEFT DoRA,
FlashAttention2 vision attention, SDPA ordinary language attention,
FlexAttention packed decoding, Hugging Face Trainer, Accelerate, DeepSpeed
ZeRO-2, pytest.

**Spec:** `docs/superpowers/specs/2026-09-04-packed-admission-design.md`

## Global Constraints

- Run Python, pytest, training, and evaluation only through
  `./venv/ilvr-venv` with `PYTHONPATH="$PWD/transformers/src:$PWD/src"`.
- Keep model and Trainer state below `checkpoints/`, results below `results/`,
  and every execution log below `logs/`; use one fresh run ID per retry.
- Keep clean `MAX_SEQUENCE_LENGTH=8192` and packed
  `STEP_MAX_PACKED_SEQUENCE_LENGTH=7168` distinct.
- Preserve DoRA rank 32, alpha 64, dropout 0.05; BF16; seed 42; language SDPA;
  vision FlashAttention2; packed FlexAttention; non-reentrant checkpointing;
  and two-rank ZeRO-2.
- Never truncate, split, prune, dynamically skip, or reorder a retained example.
- Require `tracked_worktree_clean=true` when building, verifying, capacity
  validating, launching, and evaluating. Any tracked unstaged or staged change
  fails before model execution; untracked interpreter caches remain ignored.
- The three arms share byte-identical retained source, negatives, admission
  manifest, verifier result, sampler inputs, and all settings except their loss
  coefficient triplet and run/output identities.
- Stop before full negative generation when a mandatory pilot fails. Stop
  before training when any admission, capacity, or ZeRO-2 gate fails. If a
  runtime failure occurs, preserve its fresh run log, make only the causal fix,
  add one causal regression, and retry under a new run ID.
- The user request is understood as: follow the approved v11 evidence ladder
  and continue with the smallest evidence-supported correction until the fixed
  DoRA method has a no-OOM launch path; do not silently change the scientific
  population or objective.

## Wire contracts used by every task

All JSON objects reject duplicate keys and non-finite constants. Canonical JSON
is UTF-8 with sorted object keys, compact separators, `ensure_ascii=False`, and
one final LF; SHA-256 covers those complete bytes. Stored paths are resolved
absolute paths. Readers reject symlinks/non-regular files, reopen-boundary
changes, unexpected fields, booleans where integers are required, and any hash
mismatch. Git cleanliness means `git diff --quiet` and
`git diff --cached --quiet`; untracked interpreter caches do not change the
tracked-worktree flag.

`SubsetSourceMap` schema 1 has exact fields
`schema_version,kind,original_source_sha256,subset_sha256,original_source_indices,sample_ids`.
Its kind is `stage2b-subset-source-map`; indices are unique increasing
zero-based original JSONL ordinals, sample IDs are unique nonempty strings in
the same order, and the subset bytes must equal those original lines with one LF
each. Task 7 writes `clean-source-map.json` for all 3374 clean rows and
`zero2-source-map.json` for the one global witness. Both packed-admission build
and verify require `--source-map-path` and its expected
`--source-map-sha256`; this preserves original identity even for a one-row
capacity source.

`PackedAdmissionIdentity` has exact fields
`original_data_path,original_data_sha256,clean_data_path,clean_data_sha256,source_map_path,source_map_sha256,retained_data_path,retained_data_sha256,clean_length_manifest_path,clean_length_manifest_sha256,image_root_path,checkpoint_path,checkpoint_files,tokenizer_config_identity,negative_path,negative_sha256,negative_manifest_path,negative_manifest_sha256,assistant_plan_rendering,attention_implementation,vision_attention_implementation,latent_size,clean_model_limit,clean_prefilter_limit,max_packed_sequence_length,git_commit,tracked_worktree_clean`.
`checkpoint_files` reuses every `(load_root,relative_path,size,sha256)` entry
from the strict schema-3 negative manifest and is revalidated from disk.

`PackedAdmissionRecord` has the eleven fields listed in Task 3.
`PackedTieClass` has exact fields
`packed_padded_length,vision_token_count,branch_count,scored_token_count,member_source_indices,representative_source_index`.
`PackedBoundaryWitnesses` has exact fields
`tie_classes,pareto_source_indices,global_max_source_index,final_probe_source_indices`.
Witness selection receives retained records only. A tie class is one exact
four-axis tuple `(packed_padded_length,vision_token_count,branch_count,
scored_token_count)` with at least two members; classes sort by that tuple,
members sort by source index, and the representative is the lowest member.
Pareto selection runs over exactly one representative per tuple. The global
maximum is selected independently over every retained record by
`(packed_padded_length,packed_unpadded_length,clean_length,source_index)`. Thus,
when a non-representative tie member is the global maximum, both it and the
representative intentionally appear in the final sorted union.
`PackedAdmissionCounts` has exact integer fields
`original,clean_admissible,mapped,retained,filtered`. `original` is always the
line count of `original_data_path`; `clean_admissible` is always the retained
count of the bound clean-length manifest; `mapped` is the number of records in
the supplied source map. The strict invariants are
`mapped == len(records) == retained + filtered` and
`mapped <= clean_admissible <= original`. Therefore the full build records
`3450,3374,3374,...`, while the capacity-only one-row build records
`3450,3374,1,1,0` without renumbering its original source index.
`PackedAdmissionManifest` has exact fields
`schema_version,kind,identity,records,retained_source_indices,retained_sample_ids,filtered_source_indices,filtered_sample_ids,boundary_witnesses,counts`;
schema is 1 and kind is `stage2b-packed-admission`.

`PackedAdmissionVerification` has exact fields
`schema_version,kind,status,manifest_path,manifest_sha256,clean_admissible_sha256,packed_retained_sha256,negative_sha256,negative_manifest_sha256,verified_record_count,retained_source_indices,filtered_source_indices,errors`;
kind is `stage2b-packed-admission-verification`, status is `pass|fail`, and a
failure has at least one error. Hash fields are `str | null`: when a referenced
artifact cannot be read or authenticated, that hash is null, record count is
zero, and both index arrays are empty. A manifest-path read failure still writes
this bounded fail result to the independently supplied output path. Training
accepts only a pass result with no null hash. The verifier resolves no path relative to its
invocation directory: it opens only the absolute paths frozen in the manifest.

`PilotSummary` has exact fields
`schema_version,kind,status,run_id,git_commit,tracked_worktree_clean,original_data_path,original_data_sha256,checkpoint_path,checkpoint_files,gpu,audits,admitted_source_indices,ignored_source_indices,capacity_reports,errors`.
`PackedCapacitySummary` has exact fields
`schema_version,kind,status,pilot_summary_path,pilot_summary_sha256,gpu_snapshot_path,gpu_snapshot_sha256,admission_manifest_path,admission_manifest_sha256,verification_path,verification_sha256,witness_reports,distributed_evidence_path,distributed_evidence_sha256,distributed_checkpoint_path,distributed_checkpoint_files,training_runtime_identity_path,training_runtime_identity_sha256,commands,counts,exclusions,gpu_identities,training_population,errors`.
Both use schema 1; their kinds are respectively `stage2b-packed-pilot` and
`stage2b-packed-capacity`, status is `pass|fail`, and `errors` is an array of
strings that is empty exactly on pass.
Every nested artifact reference has exact `path,sha256`; every witness report
also has `source_index,sample_id,packed_padded_length,peak_reserved_fraction,status`.
`commands` is an ordered array of command token arrays (all strings). `counts`
has exact integer fields `original,clean_admissible,mapped,retained,filtered,
boundary_witnesses`. `exclusions` is an ordered two-item array with exact fields
`stage,count,reason,source_indices,sample_ids`, for stages `clean_prefilter` and
`packed_admission`; arrays are ordered by original source index and their lengths
equal count, so all excluded IDs/reasons are retained.
`gpu_identities` is sorted by UUID and each item has exact fields
`physical_index,name,uuid,total_memory_bytes`; UUIDs are unique and include all
single-device and distributed witnesses. `training_population` is the literal
`capacity_admitted_subset`.

`PilotSummary.gpu` has exact fields
`physical_index,name,uuid,total_memory_bytes`. `audits` and `capacity_reports`
are arrays sorted by source index. Each audit has
`source_index,path,sha256,packed_padded_length,admission`; each capacity item has
`source_index,path,sha256,status,peak_reserved_fraction`. Admission is
`admitted|ignored_over_cap`.
The pilot's original source and checkpoint identity must byte-match the later
full admission identity; `checkpoint_files` uses the same complete
`(load_root,relative_path,size,sha256)` entries.

`PackedGpuSnapshot` has exact fields `schema_version,kind,gpus`; schema is 1,
kind is `stage2b-packed-gpu-snapshot`, and ordered GPU entries have exact fields
`host_physical_index,visible_cuda_index,name,uuid,total_memory_bytes`. Task 7
sets `CUDA_VISIBLE_DEVICES=0,1`, so entries are exactly host/visible `(0,0)` and
`(1,1)`, both named `NVIDIA H100 NVL` with unique UUIDs. The snapshot is captured
from one `nvidia-smi --query-gpu=index,name,uuid,memory.total` call immediately
before capacity launches; every single-device probe receives its path/hash and
must match visible CUDA 0 to entry 0, while distributed ranks 0/1 match entries
0/1 via CUDA runtime UUID/name/total memory. The final validator re-runs the same
live query and requires the snapshot still matches.

`PackedCommandLog` has exact fields `schema_version,kind,commands`; schema is 1,
kind is `stage2b-packed-command-log`, and commands is the ordered array of exact
argv string arrays for Task 7's clean build, negative generation, admission
build/verify, every witness probe, singleton build/verify/training, and capacity
evidence production. It is published exclusively before the final validator;
shell text is never reparsed to reconstruct argv. The validator invocation is
itself preserved in its execution log to avoid a self-hash cycle.

`PackedDistributedCapacityEvidence` has exact fields
`schema_version,kind,run_id,world_size,process_ranks,global_step,optimizer_step,sample_id,rank_observations,gpu_snapshot_path,gpu_snapshot_sha256,gpu_identities,checkpoint_path,training_runtime_identity_path,training_runtime_identity_sha256,method,zero_stage,finite_loss,finite_gradient,nonzero_gradient,worker_errors`.
Schema is 1, kind is `stage2b-packed-distributed-capacity`, world size is 2, ranks are
`[0,1]`, steps are both 2, method is `dora`, ZeRO stage is 2, and each ordered
rank observation has exact fields
`rank,sample_ids,pre_step_gradient_norms,training_loss`: the sample list contains
the same sample ID twice, the norm list contains exactly two finite positive
floats captured at optimizer boundaries, and loss is finite. Each ordered GPU
identity has exact fields `rank,physical_index,name,uuid,total_memory_bytes`.
The three numeric-health booleans are derived from all rank observations and
are true. It is published collectively and exclusively after training.
The capacity-only launcher exposes
`--step_packed_capacity_evidence_path` / `STEP_PACKED_CAPACITY_EVIDENCE_PATH`;
it also requires `--step_packed_gpu_snapshot`,
`--step_packed_gpu_snapshot_sha256` and the matching launcher variables
`STEP_PACKED_GPU_SNAPSHOT`, `STEP_PACKED_GPU_SNAPSHOT_SHA256`;
it is absent from normal experiment arms and is legal only for packed Stage 2B
with world size 2, ZeRO-2, `max_steps=2`, batch and accumulation 1, and a
one-record verified admission source.

Every final training artifact contains canonical
`stage2_training_runtime_identity.json` with exact fields
`schema_version,kind,run_id,git_commit,tracked_worktree_clean,coefficient_triplet,admission_manifest_sha256,verification_sha256,world_size,process_ranks,rank_observations`. Schema is 1,
kind is `stage2-training-runtime-identity`, and each rank observation has exact
fields `rank,language_attention_class,language_attention_layer_count,
vision_attention_class,vision_attention_block_count,packed_attention_backend,
sdpa_forward_counts,vision_flash_attention_forward_counts,
flex_attention_forward_counts`. Concrete class names/count rules match the
evaluation identity, packed backend is `flex_attention`, and all three counter
arrays contain positive integers, with one SDPA/Flex count per language layer
and one FlashAttention2 count per vision block. The vendored kernel methods
increment their private counters immediately before calling respectively
`scaled_dot_product_attention`, `flash_attn_varlen_func`, and
`flex_attention_forward`; observations are collectively gathered and rank zero
exclusively publishes the sidecar after training and concrete-module validation.
`tracked_worktree_clean` must be true and commit must equal the admission
identity.

Each arm's `launch-config.json` has exact fields
`schema_version,arm,run_id,coefficient_triplet,normalized_command,capacity_summary_path,capacity_summary_sha256,retained_data_sha256,negative_sha256,negative_manifest_sha256,admission_manifest_sha256,verification_sha256`.
Its schema version is 1 and arm is exactly `ce|dpo|entropy`.
The final `comparison.json` has exact fields
`schema_version,kind,primary_metric,training_population,runs`; each run has
`rank,arm,run_id,launch_config_sha256,checkpoint_sha256,training_runtime_identity_sha256,predictions_sha256,evaluation_runtime_identity_sha256,coefficient_triplet,correct,source_count,coverage_accuracy,valid_gold_accuracy,invalid_gold_count,average_inference_time_sec`.
`checkpoint_sha256` is the exact object mapping
`adapter_config.json,adapter_model.safetensors,stage2_peft_metadata.json` to
lowercase SHA-256. Commands are JSON arrays of strings and coefficient triplets
are three-number JSON arrays. Runs sort by descending coverage then arm name;
rank is competition rank `1 + count(strictly greater coverage)`, so exact ties
share rank.
Its schema version is 1, kind is `stage2b-dora-ablation-comparison`,
`primary_metric` is `coverage_accuracy`, and `training_population` is
`capacity_admitted_subset`.

Each evaluation also writes `evaluation-runtime-identity.json` with exact fields
`schema_version,kind,git_commit,tracked_worktree_clean,model_dir,language_attention_class,
language_attention_layer_count,vision_attention_class,
vision_attention_block_count`. Schema is 1, kind is
`stage2-evaluation-runtime-identity`, class names are exactly
`Qwen2_5_VLSdpaAttention` and `Qwen2_5_VLVisionFlashAttention2`, and both counts
are positive. It is written exclusively only after the loaded model's concrete
modules pass `validate_qwen2_5_vl_attention_backends`.
Its tracked flag must be true and commit must equal the admission/training
identity used for the arm.
These schemas are implemented and unit-tested in the task that first writes
them when they are durable code outputs. Task 1's deliberately pre-implementation
pilot summary and Task 8's experiment summaries use exact inline schema
validators in their publishing commands; one-off ad-hoc JSON shapes are not
accepted.

The model's clean tokenizer limit remains 8192. The clean 7168 pass in Task 7
is a lossless prefilter for a packed cap of 7168 because producer and verifier
both assert `packed_unpadded_length >= clean_length`; it does not change the
model or evaluation limit.

---

### Task 1: Run the fixed three-row tail pilot before admission code

**Files:**

- Read: `data/TRAIN.jsonl`
- Read: `checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827/`
- Write: `results/step-pref-packed-pilot-<UTC>/`
- Write: `logs/step-pref-packed-pilot-<UTC>/`

**Interfaces:**

- Consumes: existing `scripts/select_jsonl_rows.py`,
  `scripts/generate_step_negatives.py`,
  `scripts/audit_packed_stage2b_lengths.py`, and
  `scripts/probe_packed_stage2b_capacity.py`.
- Produces: three authenticated one-row artifacts, three natural-plan audits,
  capacity JSON for every admitted pilot, and `pilot-summary.json` with a
  fail-closed overall status.

- [ ] **Step 1: Reserve one fresh pilot run and verify frozen inputs/GPU**

```bash
pilot_id="step-pref-packed-pilot-$(date -u +%Y%m%dT%H%M%SZ)"
test -z "$(git diff --name-only)"
test -z "$(git diff --cached --name-only)"
test ! -e "results/$pilot_id"
test ! -e "logs/$pilot_id"
mkdir -p "results/$pilot_id" "logs/$pilot_id"
test "$(sha256sum data/TRAIN.jsonl | cut -d' ' -f1)" = 5bc1313a594f5713975b67388c745c6632275c37fd260055b27230c9cfedbed8
test "$(sha256sum checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827/config.json | cut -d' ' -f1)" = cbec86a55c91603e7f91d506d94c975113de3af4ef6e9cce6e124e6825bc7b44
test "$(sha256sum checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827/model.safetensors.index.json | cut -d' ' -f1)" = d28f59cc1060d05418f59a2320b4f53f83ba50374d7283d59ee559aefe35864b
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)"
test "$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader)" = "NVIDIA H100 NVL"
test "$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)" = 0
```

Expected: all commands exit zero before any GPU process starts.

- [ ] **Step 2: Select and generate exact negatives for rows 1372, 1714, and 1582**

```bash
set -o pipefail
for row in 1372 1714 1582; do
  ./venv/ilvr-venv/bin/python scripts/select_jsonl_rows.py \
    --input-path data/TRAIN.jsonl \
    --output-path "results/$pilot_id/row-$row.jsonl" --index "$row"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/transformers/src:$PWD/src" \
    ./venv/ilvr-venv/bin/python scripts/generate_step_negatives.py \
    --data_path "results/$pilot_id/row-$row.jsonl" --image_root data \
    --stage2a_checkpoint_path checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
    --output_path "results/$pilot_id/row-$row-negatives.jsonl" \
    --manifest_path "results/$pilot_id/row-$row-negative-manifest.json" \
    --latent_size 8 --min_tokens 2 --max_new_tokens 32 \
    --attention_implementation sdpa \
    --vision_attention_implementation flash_attention_2 --device cuda:0 \
    2>&1 | tee "logs/$pilot_id/row-$row-generate.log"
done
```

Expected: every generator exits zero and its schema-3 consumer validation
passes. Any failure rejects the pilot; do not retry inside this run ID.

- [ ] **Step 3: Reconstruct and record every natural packed plan without model execution**

```bash
for row in 1372 1714 1582; do
  PYTHONPATH="$PWD/transformers/src:$PWD/src" \
    ./venv/ilvr-venv/bin/python scripts/audit_packed_stage2b_lengths.py \
    --checkpoint_path checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
    --data_path "results/$pilot_id/row-$row.jsonl" --image_root data \
    --negative_path "results/$pilot_id/row-$row-negatives.jsonl" \
    --negative_manifest "results/$pilot_id/row-$row-negative-manifest.json" \
    --attention_implementation sdpa \
    --vision_attention_implementation flash_attention_2 \
    --report_path "results/$pilot_id/row-$row-packed-audit.json" \
    --subset_output_path "results/$pilot_id/row-$row-packed-witness.jsonl"
done
```

Expected: row 1582 is at most 7168; at least one of rows 1372/1714 is also at
most 7168. An over-cap row is recorded but not probed.

- [ ] **Step 4: Probe every admitted pilot with canonical DoRA**

```bash
set -o pipefail
for row in 1372 1714 1582; do
  padded=$(./venv/ilvr-venv/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["winner"]["padded_length"])' \
    "results/$pilot_id/row-$row-packed-audit.json")
  if (( padded > 7168 )); then
    continue
  fi
  test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)"
  test "$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)" = 0
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/transformers/src:$PWD/src" \
    ./venv/ilvr-venv/bin/python scripts/probe_packed_stage2b_capacity.py \
    --checkpoint_path checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
    --data_path "results/$pilot_id/row-$row.jsonl" --image_root data \
    --negative_path "results/$pilot_id/row-$row-negatives.jsonl" \
    --negative_manifest "results/$pilot_id/row-$row-negative-manifest.json" \
    --source_row_index "$row" --device cuda:0 --tuning_method dora \
    --attention_implementation sdpa \
    --vision_attention_implementation flash_attention_2 \
    --gradient_checkpointing \
    --output_path "results/$pilot_id/row-$row-capacity.json" \
    2>&1 | tee "logs/$pilot_id/row-$row-capacity.log"
done
```

Expected: every admitted row reports `status="pass"`, four finite losses, a
finite positive gradient norm, and `peak_reserved_bytes/device_total_bytes <=
0.85`.

- [ ] **Step 5: Publish and validate the pilot decision once**

Write `results/$pilot_id/pilot-summary.json` with the run ID, git commit, the
required `tracked_worktree_clean=true`, absolute original TRAIN path/hash,
absolute checkpoint path and every strict
schema-3 checkpoint-file identity, GPU name/UUID, three audit paths and hashes,
admitted row IDs, ignored over-cap row IDs, capacity paths and hashes, and
overall status. Use one short venv Python
command with `json.dumps(..., sort_keys=True, separators=(",", ":"),
allow_nan=False) + "\n"`; refuse an existing destination. Validate at least two
admitted rows, row 1582 admitted, every admitted report passing, and every
reserved ratio at most 0.85.

Expected: overall `pass`. Otherwise stop and return to the approved design
decision with the observed failing row and memory witness.

---

### Task 2: Add packed-admission configuration and launcher ownership

**Files:**

- Modify: `src/step_preference/config.py`
- Modify: `src/utils_deepseed.py`
- Modify: `run_stage2_training.sh`
- Modify: `tests/step_preference/test_config.py`
- Modify: `tests/stage2/test_launcher.py`

**Interfaces:**

- Produces `StepPreferenceConfig.max_packed_sequence_length: int | None` and
  four optional artifact fields:
  `packed_admission_manifest`, `packed_admission_manifest_sha256`,
  `packed_admission_verification`, and
  `packed_admission_verification_sha256`.
- Produces public CLI options
  `--step_max_packed_sequence_length`,
  `--step_packed_admission_manifest`,
  `--step_packed_admission_manifest_sha256`,
  `--step_packed_admission_verification`, and
  `--step_packed_admission_verification_sha256`.
- Launcher owns the matching `STEP_MAX_PACKED_SEQUENCE_LENGTH` (default 7168),
  `STEP_PACKED_ADMISSION_MANIFEST`,
  `STEP_PACKED_ADMISSION_MANIFEST_SHA256`,
  `STEP_PACKED_ADMISSION_VERIFICATION`, and
  `STEP_PACKED_ADMISSION_VERIFICATION_SHA256` variables.

- [ ] **Step 1: Write failing immutable-config and launcher tests**

Add tests that assert:

```python
args = parse_args(required_stage2b_packed_args + [
    "--step_max_packed_sequence_length", "7168",
    "--step_packed_admission_manifest", "admission.json",
    "--step_packed_admission_manifest_sha256", "a" * 64,
    "--step_packed_admission_verification", "verification.json",
    "--step_packed_admission_verification_sha256", "b" * 64,
])
assert args.step_preference_config.max_packed_sequence_length == 7168
assert args.step_preference_config.packed_admission_manifest == "admission.json"
```

Parametrize zero/negative cap rejection only when packing is enabled, partial
path/hash pairs, malformed SHA-256, and launcher values `0`, empty hash, or one
missing admission field. Assert ordinary and non-packed Stage 2 command tokens
remain unchanged. Assert the launcher forwards all five options and treats them
as managed.

- [ ] **Step 2: Run the focused tests to verify RED**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_config.py tests/stage2/test_launcher.py \
  -k 'packed or admission'
```

Expected: FAIL because the fields/options do not exist.

- [ ] **Step 3: Implement the minimal immutable configuration**

Add the five fields to `StepPreferenceConfig`, normalize them in `from_args`,
and validate exact path/hash pairing. `max_packed_sequence_length` defaults to
7168 at the public parser and must be a non-boolean positive integer whenever
`packing_enabled` is true. Do not require admission artifacts in the dataclass;
Task 5 enforces them only for normal packed Stage 2B training so the existing
private capacity loader can reconstruct pre-admission rows.

- [ ] **Step 4: Implement launcher defaults and fail-fast forwarding**

When `STEP_PREFERENCE_PACKING=true`, require all four admission artifact
variables for normal launcher execution, validate both hashes with
`^[0-9a-f]{64}$`, and append all five CLI options. Add them to
`MANAGED_OPTIONS`. Keep `--stage2b_preflight_only` usable with these same
artifacts. Do not create output/result/log directories before the preflight
subprocess succeeds.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_config.py tests/stage2/test_launcher.py
git add src/step_preference/config.py src/utils_deepseed.py \
  run_stage2_training.sh tests/step_preference/test_config.py \
  tests/stage2/test_launcher.py
git commit -m "feat: configure packed stage2b admission"
```

Expected: both files pass.

---

### Task 3: Implement strict packed-admission records and witness selection

**Files:**

- Create: `src/step_preference/packed_admission.py`
- Create: `tests/step_preference/test_packed_admission.py`

**Interfaces:**

- Produces frozen dataclasses `PackedAdmissionIdentity`,
  `PackedAdmissionRecord`, `PackedBoundaryWitnesses`,
  `PackedAdmissionCounts`, `PackedAdmissionManifest`, and
  `PackedAdmissionVerification`.
- Produces `select_packed_admission(records, *, max_packed_length: int) ->
  tuple[tuple[PackedAdmissionRecord, ...], tuple[PackedAdmissionRecord, ...]]`.
- Produces `select_capacity_witness_indices(records) -> tuple[int, ...]` using
  the Pareto-plus-global-maximum union from v11.
- Produces strict `read_*`, `write_*`, and SHA-256 helpers using exclusive
  canonical JSON.

- [ ] **Step 1: Write failing schema and invariant tests**

Create exact fixtures with retained and over-cap rows. Test rejection of unknown
or missing fields, booleans in integer fields, invalid hashes, duplicate source
indices/sample IDs, reordered partitions, inconsistent reasons/counts,
`unpadded < clean`, padded values not divisible by 128, a non-LF canonical
payload, symlink inputs, and attempted overwrite.

- [ ] **Step 2: Write failing Pareto/global-maximum tests**

Use records where:

```python
# source 4 is the packed tuple winner but is dominated on the four Pareto axes.
winner = record(4, padded=7168, unpadded=7150, clean=6000,
                vision=100, branches=8, scored=400)
dominator = record(3, padded=7168, unpadded=7100, clean=5900,
                   vision=120, branches=9, scored=450)
```

Assert the final set includes both the nondominated representative(s) and
source 4, is deduplicated, sorted by source index, uses the lowest source index
for exact four-axis ties, and raises when the final union contains nine rows.
Also put the global maximum on the higher-index member of an exact four-axis
tie and assert the final union retains both that member and the lower-index
Pareto representative. Assert filtered records are rejected as input to witness
selection rather than silently becoming probes.

- [ ] **Step 3: Run the new tests to verify RED**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_packed_admission.py
```

Expected: import failure because the module does not exist.

- [ ] **Step 4: Implement only the pure schema and selection layer**

Use `dataclasses.fields` for exact-field parsing and the existing canonical JSON
rules from `step_preference.negatives`. Keep filesystem/model construction out
of this task. The only admission reasons are `retained` and
`packed_length_exceeds_limit`. The global winner key is exactly
`(packed_padded_length, packed_unpadded_length, clean_length, source_index)`.
Strict dominance requires `>=` on all four Pareto axes and `>` on at least one.

```python
@dataclass(frozen=True)
class PackedAdmissionRecord:
    source_index: int
    sample_id: str
    clean_length: int
    packed_unpadded_length: int
    packed_padded_length: int
    vision_token_count: int
    image_count: int
    reasoning_block_count: int
    branch_count: int
    scored_token_count: int
    admission_reason: str

def select_packed_admission(records, *, max_packed_length):
    retained = tuple(
        row for row in records
        if row.packed_padded_length <= max_packed_length
    )
    filtered = tuple(
        row for row in records
        if row.packed_padded_length > max_packed_length
    )
    return retained, filtered

def select_capacity_witness_indices(retained_records):
    axes = lambda row: (
        row.packed_padded_length, row.vision_token_count,
        row.branch_count, row.scored_token_count,
    )
    representatives = {}
    records = tuple(retained_records)
    if any(row.admission_reason != "retained" for row in records):
        raise ValueError("capacity witnesses require retained records only")
    for row in sorted(records, key=lambda item: item.source_index):
        representatives.setdefault(axes(row), row)
    pareto = [
        row for key, row in representatives.items()
        if not any(
            all(left >= right for left, right in zip(other, key))
            and any(left > right for left, right in zip(other, key))
            for other in representatives if other != key
        )
    ]
    maximum = max(
        records,
        key=lambda row: (
            row.packed_padded_length, row.packed_unpadded_length,
            row.clean_length, row.source_index,
        ),
    )
    selected = {row.source_index: row for row in (*pareto, maximum)}
    if len(selected) > 8:
        raise ValueError("packed capacity witness set exceeds eight rows")
    return tuple(sorted(selected))
```

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_packed_admission.py
git add src/step_preference/packed_admission.py \
  tests/step_preference/test_packed_admission.py
git commit -m "feat: define packed admission manifest"
```

---

### Task 4: Build and independently verify the shared admission artifacts

**Files:**

- Modify: `src/step_preference/packed_admission.py`
- Modify: `scripts/audit_packed_stage2b_lengths.py`
- Create: `tests/step_preference/test_packed_admission_cli.py`
- Modify: `tests/step_preference/test_packed_capacity_audit.py`

**Interfaces:**

- Produces CLI `python -m step_preference.packed_admission build` with required
  `--original-data-path`, `--clean-data-path`, `--clean-length-manifest`,
  `--clean-length-manifest-sha256`, `--image-root`, `--checkpoint-path`,
  `--source-map-path`, `--source-map-sha256`, `--negative-path`,
  `--negative-manifest`,
  `--max-packed-sequence-length`, `--retained-output-path`, and
  `--manifest-output-path`.
- Produces CLI `python -m step_preference.packed_admission verify` with required
  `--manifest-path`, `--manifest-sha256`, and `--output-path`.
- Extends packed audit records with `vision_token_count`, `image_count`,
  `reasoning_block_count`, and `scored_token_count` while preserving its old
  winner and report interface.
- Producer and verifier both rebuild plans through `Stage2DataCollator` and
  `build_packed_tree_plan`; verifier never calls the producer or trusts its
  record/partition values as expected input.

- [ ] **Step 1: Write failing record-feature and build tests**

Mock only checkpoint/processor loading. Assert feature definitions:

```python
vision_token_count = int((batch["input_ids"] == model_config.image_token_id).sum())
image_count = len(prepared.original_image_paths)
reasoning_block_count = len(batch["reasoning_step_spans"])
branch_count = len(plan.branches)
scored_token_count = int(plan.selected_positions().numel())
```

Assert `build` validates the original schema-2 manifest at clean cap 7168,
requires every source-map index to belong to that manifest's retained indices,
byte-reconstructs the supplied clean JSONL from exactly the mapped original
indices, validates
schema-3 negatives against the clean source, writes the retained JSONL in
original order, records all 3374 clean-admissible rows, and binds every input
hash plus git commit and required `tracked_worktree_clean=true`. A dirty tracked
tree makes both build and verify fail without publishing a pass artifact. Add a
one-row source-map fixture whose
original index is 1372 and assert both its admission record and retained index
remain 1372 rather than being renumbered to zero.

- [ ] **Step 2: Write failing independent-verifier tests**

Mutate one field at a time after producing a valid fixture: packed length,
vision count, reason, retained order, retained JSONL byte, negative-manifest
hash, backend, cap, and git identity. Assert `verify` writes status `fail`, exits
nonzero, and never changes the admission manifest. A valid fixture writes one
canonical pass result containing the manifest, clean source, retained source,
negative, and negative-manifest hashes.

Also make the manifest itself unreadable or hash-mismatched and assert the
verifier publishes its exact bounded fail shape: `manifest_sha256` and all other
untrusted artifact hashes are null, count zero, empty index arrays, one error,
and exit 2. An observed hash that disagrees with the expected CLI hash is never
published as authenticated identity.

- [ ] **Step 3: Run tests to verify RED**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_packed_admission_cli.py \
  tests/step_preference/test_packed_capacity_audit.py
```

Expected: FAIL on missing build/verify behavior and record features.

- [ ] **Step 4: Implement shared low-level reconstruction, separate decisions**

Extract a read-only `reconstruct_natural_packed_records(...)` primitive that
returns immutable measured records. `build` applies admission and witness
selection and writes the two exclusive outputs. `verify` independently reloads
all bound paths/hashes, invokes reconstruction, computes expected partitions,
and compares complete dataclasses. It must not call `build_manifest`, copy the
manifest's records as expectations, or repair any artifact.

```python
def build_packed_admission(args):
    clean = validate_clean_source_chain(args)
    records = reconstruct_natural_packed_records(
        checkpoint_path=args.checkpoint_path,
        data_path=args.clean_data_path,
        image_root=args.image_root,
        negative_path=args.negative_path,
        negative_manifest=args.negative_manifest,
        original_source_indices=clean.original_source_indices,
        max_packed_length=args.max_packed_sequence_length,
    )
    retained, filtered = select_packed_admission(
        records, max_packed_length=args.max_packed_sequence_length
    )
    manifest = assemble_manifest(args, clean, records, retained, filtered)
    write_exact_source_subset(
        args.clean_data_path, args.retained_output_path,
        retained_clean_offsets(clean, retained),
    )
    write_packed_admission_manifest(args.manifest_output_path, manifest)
    return manifest

def verify_packed_admission(args):
    manifest = read_and_authenticate_manifest(
        args.manifest_path, args.manifest_sha256
    )
    observed = independently_reconstruct_from_identity(manifest.identity)
    verification = compare_complete_admission(manifest, observed)
    write_packed_admission_verification(args.output_path, verification)
    return verification
```

- [ ] **Step 5: Run focused regressions and commit**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_packed_admission.py \
  tests/step_preference/test_packed_admission_cli.py \
  tests/step_preference/test_packed_capacity_audit.py \
  tests/step_preference/test_select_jsonl_rows.py
git add src/step_preference/packed_admission.py \
  scripts/audit_packed_stage2b_lengths.py \
  tests/step_preference/test_packed_admission_cli.py \
  tests/step_preference/test_packed_capacity_audit.py
git commit -m "feat: build and verify packed admission"
```

---

### Task 5: Enforce admission in normal packed Stage 2B training

**Files:**

- Modify: `src/main.py`
- Modify: `src/stage2/data.py`
- Modify: `src/step_preference/equivalence.py`
- Modify: `tests/step_preference/test_stage2b_artifacts.py`
- Modify: `tests/step_preference/test_packed_capacity_probe.py`
- Modify: `tests/stage2/test_data.py`

**Interfaces:**

- Extends `Stage2BPreflight` with the validated admission snapshot and retained
  record map when packing is enabled.
- Normal packed Stage 2B preflight requires the four admission artifact fields,
  verifies their supplied SHA-256 values, validates negatives against the
  manifest-bound clean-admissible source, and validates `args.data_path` as the
  packed-retained source.
- `Stage2DataCollator` accepts optional immutable
  `packed_admission_records_by_sample_id`; after rebuilding the natural plan it
  raises before model forward if sample ID, unpadded length, padded length, or
  cap differs.
- `load_production_context` sets private attribute
  `args._stage2b_capacity_probe = True` after parsing. No public option exposes
  this bypass; normal `main_train` rejects missing admission artifacts.
- Extends `load_production_context(..., requested_sample_id: str | None = None)`;
  `None` preserves the lowest-valid-pair behavior and a supplied ID selects
  exactly that authenticated clean-source sample.

- [ ] **Step 1: Write failing preflight source-chain tests**

Build tiny original, clean, retained, negative, admission, and verifier fixtures.
Assert normal packed preflight succeeds only when:

```text
original --clean manifest--> clean-admissible --admission--> packed-retained
clean-admissible --negative manifest--> shared negatives
```

Mutate each path/hash/order and assert failure before model construction.
Assert non-packed Stage 2B and ordinary Stage 2 behavior remain unchanged.

- [ ] **Step 2: Write failing runtime-collation and private-bypass tests**

Assert a matching retained sample collates unchanged. Monkeypatch the planner to
change padded length or unpadded length and assert the collator raises before
return. Assert an unrecorded sample raises. Assert `load_production_context`
still constructs one pilot without admission, selects an explicitly requested
sample ID from a multi-row authenticated clean source, rejects a missing ID,
and leaves the default lowest-valid behavior unchanged. A public parsed packed
training invocation without admission must fail.

- [ ] **Step 3: Run tests to verify RED**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_stage2b_artifacts.py \
  tests/step_preference/test_packed_capacity_probe.py \
  tests/stage2/test_data.py -k 'admission or capacity_probe or packed'
```

- [ ] **Step 4: Implement the smallest fail-closed integration**

Keep negative validation rooted at the clean source recorded by the admission
manifest, then select prepared examples by the retained sample-ID order and
cross-check the packed-retained JSONL bytes/hash. Pass only retained records to
the collator. Do not dynamically filter in the dataloader. The private capacity
attribute is set only inside `load_production_context`; reject it if supplied by
ordinary `parse_args` or shell arguments.

```python
if config.packing_enabled and not getattr(args, "_stage2b_capacity_probe", False):
    admission = validate_packed_admission_inputs(
        manifest_path=config.packed_admission_manifest,
        manifest_sha256=config.packed_admission_manifest_sha256,
        verification_path=config.packed_admission_verification,
        verification_sha256=config.packed_admission_verification_sha256,
        packed_data_path=args.data_path,
    )
else:
    admission = None

collator = Stage2DataCollator(
    processor, token_ids, decision.latent_size,
    decision.supervise_latent_boundaries,
    step_preference_enabled=config.is_stage2b,
    negative_store=negative_store,
    packed_admission_records_by_sample_id=(
        admission.records_by_sample_id if admission is not None else None
    ),
)
```

- [ ] **Step 5: Run regressions and commit**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_stage2b_artifacts.py \
  tests/step_preference/test_packed_capacity_probe.py \
  tests/step_preference/test_stage2b_trainer.py \
  tests/stage2/test_data.py tests/stage2/test_launcher.py
git add src/main.py src/stage2/data.py src/step_preference/equivalence.py \
  tests/step_preference/test_stage2b_artifacts.py \
  tests/step_preference/test_packed_capacity_probe.py tests/stage2/test_data.py
git commit -m "fix: enforce packed admission before stage2b"
```

---

### Task 6: Add bounded capacity acceptance evidence

**Files:**

- Create: `src/step_preference/capacity_evidence.py`
- Modify: `src/main.py`
- Modify: `src/stage2/data.py`
- Modify: `src/stage2/trainer.py`
- Modify: `src/utils_deepseed.py`
- Modify: `run_stage2_training.sh`
- Modify: `eval.py`
- Modify: `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
- Modify: `scripts/probe_packed_stage2b_capacity.py`
- Create: `scripts/validate_packed_capacity_gate.py`
- Modify: `tests/step_preference/test_packed_capacity_probe.py`
- Create: `tests/step_preference/test_packed_capacity_evidence.py`
- Create: `tests/step_preference/test_packed_capacity_gate.py`
- Modify: `tests/stage2/test_launcher.py`
- Modify: `tests/stage2/test_eval_integration.py`
- Modify: `tests/stage2/test_attention_backend.py`

**Interfaces:**

- Capacity JSON records GPU name/UUID, physical index, and
  `peak_reserved_fraction`; status is pass only when the fraction is at most the
  CLI `--max_peak_reserved_fraction` default 0.85.
- The probe CLI accepts optional `--sample_id`; when present it passes that ID
  to `load_production_context` and requires the resulting batch/report identity
  to match it.
- The probe also requires `--gpu-snapshot` and `--gpu-snapshot-sha256`; the
  capacity-only launcher owns matching
  `STEP_PACKED_GPU_SNAPSHOT` / `STEP_PACKED_GPU_SNAPSHOT_SHA256` and forwards
  them with the distributed evidence path.
- `validate_packed_capacity_gate.py` consumes one verified admission manifest,
  one capacity JSON per final witness, and one distributed training evidence
  JSON; it writes one exclusive `capacity-summary.json` and exits zero only if
  all exact identities and gates pass.
- The validator CLI requires `--admission-manifest`,
  `--admission-manifest-sha256`, `--pilot-summary`,
  `--pilot-summary-sha256`, `--gpu-snapshot`, `--gpu-snapshot-sha256`,
  `--commands-path`, `--commands-sha256`,
  `--verification`, `--verification-sha256`, one or more repeated
  `--capacity-report`, `--distributed-evidence`, `--distributed-checkpoint`, and
  `--output-path`.
- The new distributed evidence publisher is deliberately separate from the
  existing four-rank benchmark evidence protocol. It gathers exact sample IDs
  observed by both ranks, authenticates the final output-root artifact saved
  after step 2, and writes one `PackedDistributedCapacityEvidence` on rank zero
  after all ranks finish.
- Every capacity GPU must have name exactly `NVIDIA H100 NVL`. Immediately
  before launch, bind each visible physical index to its live `nvidia-smi` UUID;
  the report/evidence name, UUID, index, and total-memory bytes must match that
  live snapshot. Apply this predicate to every single-device witness and both
  distributed ranks, not only the pilot.

- [ ] **Step 1: Write failing headroom and identity tests**

Extend the fake runtime with `device_name()` and `device_uuid()`. Test exact
ratios 0.85 (pass) and greater than 0.85 (error), non-H100 identity, non-finite
loss, zero gradient, mismatched sample/length, duplicate evidence paths, and an
existing output. Test that the final witness set equals the manifest's complete
Pareto-plus-maximum set rather than a caller-selected subset.

Add failing publisher/launcher tests for the exact two-rank schema: each rank
must observe the sole admitted sample exactly twice; `global_step` and
`optimizer_step` must both be 2; loss and gradient must be finite/nonzero; all
worker error arrays must be empty; and only rank zero may publish after the
collective gather. Reject world sizes other than two, any different sample,
partial/malformed evidence path configuration, a non-ZeRO-2 run, or a missing
final output-root Stage 2 DoRA artifact.
Test a swapped rank/visible index, stale UUID, wrong total memory, changed live
snapshot, and missing snapshot hash; each fails before model execution or final
capacity publication.
Also mutate the pilot status, original-source hash, or any checkpoint-file hash
and assert the final gate rejects it against the admission identity.

In `test_stage2_trainer.py`, drive the existing DeepSpeed step adapter across two
true accumulation boundaries and assert it records the pre-step norm both times,
before the wrapped optimizer step clears gradients. Assert a non-boundary call
records nothing and that non-finite/zero observations fail the capacity
publisher.

In `test_eval_integration.py`, load a fake resolved Qwen with multiple concrete
language/vision blocks and require an exclusive canonical runtime-identity
sidecar only after backend validation. Mutate one module class and assert no
pass sidecar is written.

In `test_attention_backend.py`, execute language SDPA, language Flex, and vision
FlashAttention2 methods with their kernels stubbed and assert each module-local
counter increments immediately before its stub. In trainer/integration tests,
require the final training runtime sidecar to list positive SDPA/Flex counts per
actual language layer and positive FlashAttention2 counts per vision block on
every rank, exact concrete class names, and packed backend `flex_attention`;
ordinary or non-packed runs may omit this sidecar.

- [ ] **Step 2: Run tests to verify RED**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_packed_capacity_probe.py \
  tests/step_preference/test_packed_capacity_evidence.py \
  tests/step_preference/test_stage2b_trainer.py \
  tests/stage2/test_launcher.py \
  tests/stage2/test_eval_integration.py \
  tests/stage2/test_attention_backend.py \
  tests/step_preference/test_packed_capacity_gate.py
```

- [ ] **Step 3: Implement the report additions and one validator**

Reset peaks immediately before `Stage2BTrainingStep`, retain the baseline in the
reported peak, synchronize after backward, and compute the ratio from CUDA total
memory. The validator reads once, rejects duplicate JSON keys/non-finite
constants, authenticates all files by SHA-256, and publishes only one summary;
do not add retries or a recovery layer.

Wrap the capacity-only collator with a minimal recorder in
`capacity_evidence.py`; with workers fixed to zero it appends the returned
batch's authenticated `sample_id` on each rank. Extend the already-installed
`_Stage2DeepSpeedGradientAdapter`: at every true accumulation boundary, after
`collect_deepspeed_pre_step_norm` observes the reduced/unscaled ZeRO-2 gradient
and immediately before `original_step`, append that float to
`trainer._observed_pre_step_gradient_norms`. No post-training parameter-gradient
scan is evidence for this gate. After `trainer.train()` has completed step 2,
save the final model and processor collectively and wait for all ranks; only
then publish the collective training-runtime identity, and only then collect
each rank's `(rank,sample_ids,pre_step_gradient_norms,training_loss,
gpu_identity,error)` with `torch.distributed.all_gather_object`. Every rank
requires exactly two positive finite norms, two matching sample observations,
one finite loss, and no worker error; derive the three health booleans from
those predicates. Have rank zero write the exclusive evidence file, then
barrier.
Wire this call from `run_stage2` only when the dedicated capacity-evidence option
is set. Do not call or relax `stage2.evidence`, whose established protocol
requires four ranks.

For this capacity-only path, reorder `run_stage2` so the existing final-model
save and processor publication complete before capacity evidence. The evidence
`checkpoint_path` is the absolute `args.save_model_path` output root—not its
numbered `checkpoint-2` child—and `inspect_stage2_artifact` must classify that
root as the final Stage 2 DoRA artifact. `global_step=optimizer_step=2` proves
the artifact was produced after two completed optimizer steps; no identity
equivalence with `checkpoint-2` is asserted.

For final gate validation, decode the exact schema rather than accepting loose
dictionaries. Re-run `inspect_stage2_artifact` on the supplied distributed
checkpoint, require Stage 2 DoRA, locate and authenticate exactly
`adapter_config.json`, `adapter_model.safetensors`, and
`stage2_peft_metadata.json`, and store their `(relative_path,size,sha256)` in
`distributed_checkpoint_files`. Authenticate every witness report against the
manifest's final witness set and order, and require the distributed evidence
sample to equal the manifest global maximum.

Add required eval CLI `--runtime_identity_path`. After model loading and
concrete-module validation, count the actual language and vision attention
instances and exclusively write the exact evaluation runtime schema above.
This code lands before Task 7 so the admission manifest, capacity evidence, and
all three arms bind the same Git commit.

Initialize private zero counters on each vendored attention module and increment
them immediately before the actual SDPA, FlexAttention, and vision
FlashAttention2 kernel calls.
After packed Stage 2B training, unwrap/resolve the Qwen model, validate all
concrete SDPA/FA2 modules, require every language-layer counter positive, and
collect the exact observation from every rank before rank zero exclusively
publishes `stage2_training_runtime_identity.json` in the final output root.
Bind its run ID to the output-root name, its coefficient triplet to the parsed
training args, and its admission/verifier hashes to the already authenticated
StepPreferenceConfig values.
Capacity evidence authenticates this sidecar by path/hash; normal arm artifact
validation requires it as well.

```python
peak_reserved = int(runtime.max_memory_reserved())
total = int(runtime.device_total_bytes())
fraction = peak_reserved / total
report["memory"]["peak_reserved_fraction"] = fraction
report["gpu"] = {
    "physical_index": runtime.physical_index(),
    "name": runtime.device_name(),
    "uuid": runtime.device_uuid(),
    "total_memory_bytes": total,
}
if fraction > args.max_peak_reserved_fraction:
    report["status"] = "error"
    report["errors"] = ["peak reserved memory exceeds the accepted fraction"]
```

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_packed_capacity_probe.py \
  tests/step_preference/test_packed_capacity_evidence.py \
  tests/step_preference/test_stage2b_trainer.py \
  tests/stage2/test_launcher.py \
  tests/stage2/test_eval_integration.py \
  tests/stage2/test_attention_backend.py \
  tests/step_preference/test_packed_capacity_gate.py
git add src/step_preference/capacity_evidence.py src/main.py src/stage2/data.py \
  src/stage2/trainer.py \
  src/utils_deepseed.py run_stage2_training.sh eval.py \
  transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py \
  scripts/probe_packed_stage2b_capacity.py \
  scripts/validate_packed_capacity_gate.py \
  tests/step_preference/test_packed_capacity_probe.py \
  tests/step_preference/test_packed_capacity_evidence.py \
  tests/step_preference/test_stage2b_trainer.py \
  tests/stage2/test_launcher.py \
  tests/stage2/test_eval_integration.py \
  tests/stage2/test_attention_backend.py \
  tests/step_preference/test_packed_capacity_gate.py
git commit -m "test: validate packed dora capacity"
```

---

### Task 7: Materialize and probe the full capacity-admitted cohort

**Files:**

- Write: `results/step-pref-packed-admission-<UTC>/`
- Write: `logs/step-pref-packed-admission-<UTC>/`
- Write: `checkpoints/step-pref-packed-admission-<UTC>-zero2/`

**Interfaces:**

- Consumes: Tasks 1-6, original TRAIN, frozen Stage 2A, and the v11 cap.
- Produces: clean manifest/subset, one shared schema-3 negative artifact,
  admission manifest/retained source, independent verifier result, all boundary
  capacity probes, two-step ZeRO-2 checkpoint/evidence, and one pass summary.

- [ ] **Step 1: Run one focused implementation suite and allocate a fresh ID**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference tests/stage2
test -z "$(git diff --name-only)"
test -z "$(git diff --cached --name-only)"
pilot_summary="$(realpath -- "${PILOT_SUMMARY_PATH:?Task 1 pilot path required}")"
pilot_summary_sha=$(sha256sum "$pilot_summary" | cut -d' ' -f1)
admission_id="step-pref-packed-admission-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "results/$admission_id"
test ! -e "logs/$admission_id"
test ! -e "checkpoints/$admission_id-zero2"
mkdir -p "results/$admission_id" "logs/$admission_id"
```

Use one venv Python validator to require the selected pilot summary's exact
schema/status pass, frozen original TRAIN hash, complete frozen checkpoint-file
hashes, `tracked_worktree_clean=true`, and H100 NVL identity before allocating
the ID. Expected: focused suite
passes once, tracked tree is clean, pilot chain authenticates, and paths are
fresh.
From this point, construct each producing command as an argv array in the venv
helper before executing it and append that exact array to the in-memory Task 7
command list; do not reconstruct commands from shell logs later.

- [ ] **Step 2: Build the clean-7168 manifest and exact subset**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  -m stage2.length_manifest \
  --data-path data/TRAIN.jsonl --image-root data \
  --processor-source checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
  --latent-size 8 --supervise-latent-boundaries \
  --max-sequence-length 7168 \
  --assistant-plan-rendering adjacent-text-space-lf-v1 \
  --manifest-path "results/$admission_id/clean-length-manifest.json"
```

Use one venv Python command to read the strict manifest, require `3374`
retained rows and `76` filtered rows, and call
`scripts.select_jsonl_rows.select_jsonl_rows` with its retained indices to
publish `clean-admissible.jsonl`. In the same command, publish canonical
`clean-source-map.json` with the original TRAIN hash, subset hash, exact
retained indices, and canonical prepared sample IDs. Record all three SHA-256
values.

- [ ] **Step 3: Generate the shared negatives exactly once**

```bash
set -o pipefail
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/python scripts/generate_step_negatives.py \
  --data_path "results/$admission_id/clean-admissible.jsonl" --image_root data \
  --stage2a_checkpoint_path checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
  --output_path "results/$admission_id/negatives.jsonl" \
  --manifest_path "results/$admission_id/negative-manifest.json" \
  --latent_size 8 --min_tokens 2 --max_new_tokens 32 \
  --attention_implementation sdpa \
  --vision_attention_implementation flash_attention_2 --device cuda:0 \
  2>&1 | tee "logs/$admission_id/generate.log"
```

Expected: one all-or-nothing schema-3 artifact for 3374 source rows.

- [ ] **Step 4: Build and verify packed admission**

Run `python -m step_preference.packed_admission build` with the paths from Steps
2-3, original TRAIN, image root, checkpoint, cap 7168, and outputs
`packed-retained.jsonl` and `packed-admission.json`. Then hash the manifest and
run `verify` to `packed-admission-verification.json`; hash that result. Require
verifier status pass, nonempty retained rows, `filtered < 34`, and final witness
count at most eight. Tee build/verify logs under `logs/$admission_id/`.

```bash
clean_manifest_sha=$(sha256sum "results/$admission_id/clean-length-manifest.json" | cut -d' ' -f1)
source_map_sha=$(sha256sum "results/$admission_id/clean-source-map.json" | cut -d' ' -f1)
set -o pipefail
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  -m step_preference.packed_admission build \
  --original-data-path data/TRAIN.jsonl \
  --clean-data-path "results/$admission_id/clean-admissible.jsonl" \
  --clean-length-manifest "results/$admission_id/clean-length-manifest.json" \
  --clean-length-manifest-sha256 "$clean_manifest_sha" --image-root data \
  --source-map-path "results/$admission_id/clean-source-map.json" \
  --source-map-sha256 "$source_map_sha" \
  --checkpoint-path checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
  --negative-path "results/$admission_id/negatives.jsonl" \
  --negative-manifest "results/$admission_id/negative-manifest.json" \
  --max-packed-sequence-length 7168 \
  --retained-output-path "results/$admission_id/packed-retained.jsonl" \
  --manifest-output-path "results/$admission_id/packed-admission.json" \
  2>&1 | tee "logs/$admission_id/packed-admission-build.log"
admission_sha=$(sha256sum "results/$admission_id/packed-admission.json" | cut -d' ' -f1)
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  -m step_preference.packed_admission verify \
  --manifest-path "results/$admission_id/packed-admission.json" \
  --manifest-sha256 "$admission_sha" \
  --output-path "results/$admission_id/packed-admission-verification.json" \
  2>&1 | tee "logs/$admission_id/packed-admission-verify.log"
verification_sha=$(sha256sum "results/$admission_id/packed-admission-verification.json" | cut -d' ' -f1)
```

- [ ] **Step 5: Probe every final boundary witness sequentially**

Immediately before the first probe, require both GPUs idle and use one venv
Python command around a single `nvidia-smi` query to exclusively publish
canonical `gpu-snapshot.json` with the exact two-GPU mapping above; record
`gpu_snapshot_sha`. Before every later witness and distributed launch, re-query
and require the same UUID/name/total mapping. Set `CUDA_VISIBLE_DEVICES=0` for
each single-device probe and pass `--gpu-snapshot` plus its SHA-256.

For each manifest witness, run the canonical DoRA capacity probe against the
shared `clean-admissible.jsonl` and full shared negative artifact, passing the
recorded `--sample_id` so `load_production_context` selects exactly that row.
Run it on idle physical GPU 0 and store
`witness-<source-index>-capacity.json` and its log. Require status pass and peak
reserved fraction at most 0.85 before continuing to the next witness.

- [ ] **Step 6: Complete two optimizer steps under exact two-rank ZeRO-2**

Use `scripts/select_jsonl_rows.py` with the global winner's original zero-based
index to create `zero2-witness.jsonl`, then generate its exact one-row schema-3
negative artifact. Publish `zero2-source-map.json` with the full original TRAIN
hash, the one-row subset hash, the global original index, and sample ID. Reuse
the full `clean-length-manifest.json`; then invoke `packed_admission build` with
`--original-data-path data/TRAIN.jsonl`, `--clean-data-path
zero2-witness.jsonl`, the one-row source map and negative pair, cap 7168, and outputs
`zero2-packed-retained.jsonl` and `zero2-packed-admission.json`. Invoke
`packed_admission verify` with its computed hash and output
`zero2-packed-admission-verification.json`. Require its natural plan fields to
equal the corresponding full admission record. This
capacity-only copy does not replace the shared training artifacts.

Launch that one-row chain through `run_stage2_training.sh` with
`STAGE2_NUM_PROCESSES=2`, batch 1, accumulation 1, `--max_steps 2 --epochs 1`,
combined-positive `(1,1,0.1)`, DoRA 32/64/0.05, checkpointing, SDPA/FA2, its
one-row packed cap/admission/verifier paths and hashes, and
`configs/deepspeed_zero2.json`. Require both ranks to record the same witness at
steps 1 and 2, finite losses, no skipped update/OOM, and a valid Stage 2 DoRA
artifact under `checkpoints/$admission_id-zero2/`.

This is the approved v11 practical gate, not a mathematical memory upper bound.
With accumulation 1, optimizer state is allocated at step 1 and remains present
during the identical-shape step-2 microbatch; final accumulation 2 has the same
per-microbatch packed activation shape and fewer optimizer boundaries. The
remaining non-monotonic allocator uncertainty is the explicit v11 risk waiver;
do not silently change this gate to a different experiment.

```bash
zero2_admission_sha=$(sha256sum "results/$admission_id/zero2-packed-admission.json" | cut -d' ' -f1)
zero2_verification_sha=$(sha256sum "results/$admission_id/zero2-packed-admission-verification.json" | cut -d' ' -f1)
CUDA_VISIBLE_DEVICES=0,1 \
STAGE1_CHECKPOINT="$PWD/checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827" \
STAGE2A_CHECKPOINT_PATH="$PWD/checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827" \
DATA_PATH="$PWD/results/$admission_id/zero2-packed-retained.jsonl" \
IMAGE_ROOT="$PWD/data" OUTPUT_DIR="$PWD/checkpoints/$admission_id-zero2" \
LOG_FILE="$PWD/logs/$admission_id/zero2.log" \
STEP_PACKED_CAPACITY_EVIDENCE_PATH="$PWD/results/$admission_id/zero2-evidence.json" \
STEP_PACKED_GPU_SNAPSHOT="$PWD/results/$admission_id/gpu-snapshot.json" \
STEP_PACKED_GPU_SNAPSHOT_SHA256="$gpu_snapshot_sha" \
STEP_PREFERENCE_ENABLED=true STEP_PREFERENCE_MODE=stage2b \
STEP_PREFERENCE_PACKING=true STEP_MAX_PACKED_SEQUENCE_LENGTH=7168 \
STEP_NEGATIVE_PATH="$PWD/results/$admission_id/zero2-negatives.jsonl" \
STEP_NEGATIVE_MANIFEST="$PWD/results/$admission_id/zero2-negative-manifest.json" \
STEP_PACKED_ADMISSION_MANIFEST="$PWD/results/$admission_id/zero2-packed-admission.json" \
STEP_PACKED_ADMISSION_MANIFEST_SHA256="$zero2_admission_sha" \
STEP_PACKED_ADMISSION_VERIFICATION="$PWD/results/$admission_id/zero2-packed-admission-verification.json" \
STEP_PACKED_ADMISSION_VERIFICATION_SHA256="$zero2_verification_sha" \
CE_WEIGHT=1 STEP_DPO_WEIGHT=1 STEP_ENTROPY_WEIGHT=0.1 \
STEP_DPO_BETA=0.1 STEP_ENTROPY_MARGIN=0.1 \
TUNING_METHOD=dora LORA_R=32 LORA_ALPHA=64 LORA_DROPOUT=0.05 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
DEEPSPEED_CONFIG="$PWD/configs/deepspeed_zero2.json" LEARNING_RATE=5e-5 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=1 GRADIENT_CHECKPOINTING=true \
ATTENTION_IMPLEMENTATION=sdpa VISION_ATTENTION_IMPLEMENTATION=flash_attention_2 \
MAX_SEQUENCE_LENGTH=8192 SUPERVISE_LATENT_BOUNDARIES=true \
DATALOADER_NUM_WORKERS=0 STAGE2_NUM_PROCESSES=2 \
./run_stage2_training.sh --max_steps 2 --epochs 1 --save_steps 2
```

- [ ] **Step 7: Publish the capacity decision**

Run `scripts/validate_packed_capacity_gate.py` over the verified admission,
every single-device capacity JSON, and the two-rank evidence. Require overall
pass. Record all commands, hashes, counts, exclusions, GPU identities, and the
capacity-admitted-subset limitation in
`results/$admission_id/capacity-summary.json`.
First exclusively publish the exact `packed-command-log.json` from the collected
argv arrays and compute `commands_sha`.

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  scripts/validate_packed_capacity_gate.py \
  --pilot-summary "$pilot_summary" \
  --pilot-summary-sha256 "$pilot_summary_sha" \
  --gpu-snapshot "results/$admission_id/gpu-snapshot.json" \
  --gpu-snapshot-sha256 "$gpu_snapshot_sha" \
  --commands-path "results/$admission_id/packed-command-log.json" \
  --commands-sha256 "$commands_sha" \
  --admission-manifest "results/$admission_id/packed-admission.json" \
  --admission-manifest-sha256 "$admission_sha" \
  --verification "results/$admission_id/packed-admission-verification.json" \
  --verification-sha256 "$verification_sha" \
  "${capacity_report_args[@]}" \
  --distributed-evidence "results/$admission_id/zero2-evidence.json" \
  --distributed-checkpoint "checkpoints/$admission_id-zero2" \
  --output-path "results/$admission_id/capacity-summary.json"
```

---

### Task 8: Train and evaluate CE-only, DPO-only, and entropy-only

**Files:**

- Write: `checkpoints/step-pref-dora-{ce,dpo,entropy}-<UTC>/`
- Write: `results/step-pref-dora-{ce,dpo,entropy}-<UTC>/`
- Write: `logs/step-pref-dora-{ce,dpo,entropy}-<UTC>/`
- Write: `results/step-pref-dora-comparison-<UTC>/`

**Interfaces:**

- Consumes: Task 7 pass summary and its shared retained source, negatives,
  admission manifest, verifier result, and hashes.
- Produces: three complete two-epoch DoRA artifacts, three 400-row evaluations,
  and one comparison ranked by coverage accuracy.

- [ ] **Step 1: Freeze normalized launch records**

Require the tracked tree clean and current commit equal the Task 7 admission and
training-runtime identities before creating the ablation stamp.

Use one UTC stamp and write one canonical `launch-config.json` per arm. Normalize
only run ID, checkpoint/result/log paths, and coefficient triplet; assert every
other field is identical. Triplets are CE `(1,0,0)`, DPO `(0,1,0)`, and entropy
`(0,0,1)`. Bind the Task 7 capacity summary and all shared artifact hashes.
Compute and retain each exclusive launch config's SHA-256 before its real launch.

Use this one launcher function for dry-run normalization and all real arms:

```bash
ablation_stamp=$(date -u +%Y%m%dT%H%M%SZ)
shared="$PWD/results/$admission_id"
launch_arm() {
  arm=$1; ce=$2; dpo=$3; entropy=$4
  run_id="step-pref-dora-$arm-$ablation_stamp"
  STAGE1_CHECKPOINT="$PWD/checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827" \
  STAGE2A_CHECKPOINT_PATH="$PWD/checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827" \
  DATA_PATH="$shared/packed-retained.jsonl" IMAGE_ROOT="$PWD/data" \
  OUTPUT_DIR="$PWD/checkpoints/$run_id" LOG_FILE="$PWD/logs/$run_id/train.log" \
  STEP_PREFERENCE_ENABLED=true STEP_PREFERENCE_MODE=stage2b \
  STEP_PREFERENCE_PACKING=true STEP_MAX_PACKED_SEQUENCE_LENGTH=7168 \
  STEP_NEGATIVE_PATH="$shared/negatives.jsonl" \
  STEP_NEGATIVE_MANIFEST="$shared/negative-manifest.json" \
  STEP_PACKED_ADMISSION_MANIFEST="$shared/packed-admission.json" \
  STEP_PACKED_ADMISSION_MANIFEST_SHA256="$admission_sha" \
  STEP_PACKED_ADMISSION_VERIFICATION="$shared/packed-admission-verification.json" \
  STEP_PACKED_ADMISSION_VERIFICATION_SHA256="$verification_sha" \
  CE_WEIGHT="$ce" STEP_DPO_WEIGHT="$dpo" STEP_ENTROPY_WEIGHT="$entropy" \
  STEP_DPO_BETA=0.1 STEP_ENTROPY_MARGIN=0.1 \
  TUNING_METHOD=dora LORA_R=32 LORA_ALPHA=64 LORA_DROPOUT=0.05 \
  LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  DEEPSPEED_CONFIG="$PWD/configs/deepspeed_zero2.json" LEARNING_RATE=5e-5 \
  PER_DEVICE_TRAIN_BATCH_SIZE=1 \
  GRADIENT_ACCUMULATION_STEPS=2 GRADIENT_CHECKPOINTING=true \
  ATTENTION_IMPLEMENTATION=sdpa VISION_ATTENTION_IMPLEMENTATION=flash_attention_2 \
  MAX_SEQUENCE_LENGTH=8192 SUPERVISE_LATENT_BOUNDARIES=true \
  DATALOADER_NUM_WORKERS=0 STAGE2_NUM_PROCESSES=2 \
  ./run_stage2_training.sh --epochs 2 --warm_up_steps 10 --save_steps 612
}
```

Invoke it with `STAGE2_DRY_RUN=true` for all three triplets. Parse each emitted
command with `shlex.split`, replace only the values following
`--save_model_path`, `--log_file`, `--stage2_length_manifest`, `--ce_weight`,
`--step_dpo_weight`, and `--step_entropy_weight`, and require the normalized
token lists to be equal before any real launch.
After replacement, fail if any normalized token still contains one of the three
run IDs. No other option or positional token may vary.

- [ ] **Step 2: Launch CE-only and validate its final artifact**

Invoke `run_stage2_training.sh` with the Task 7 packed-retained source and shared
artifacts, `TUNING_METHOD=dora`, rank/alpha/dropout 32/64/0.05,
`GRADIENT_ACCUMULATION_STEPS=2`, `MAX_SEQUENCE_LENGTH=8192`, packed cap 7168,
checkpointing true, SDPA/FA2, ZeRO-2, seed 42, two epochs, and CE coefficients
`1,0,0`. Require exit zero, finite metrics, no skipped update/OOM, and
`inspect_stage2_artifact` classification Stage 2 DoRA. Authenticate the final
training-runtime sidecar and require actual SDPA/FA2 classes plus a positive
FlexAttention count for every language layer.

```bash
launch_arm ce 1 0 0
```

- [ ] **Step 3: Launch DPO-only, then entropy-only, with the same validation**

Run sequentially, changing only the run identity and coefficient triplet to
`0,1,0` and then `0,0,1`. A failed arm stops later arms. Do not resume a partial
run; diagnose, add one causal test/fix, and use a fresh run ID.

The required causal regression/fix changes the tracked Git identity, so any arm
failure invalidates the Task 7 capacity summary and every arm. Commit the causal
test/fix, rerun Task 7 under a fresh admission ID and new ablation stamp, then
rerun CE/DPO/entropy from the beginning; never mix an earlier successful arm
with post-fix arms. Any proposed change to cap, population, objective, or
comparison semantics is outside commitment v11 and must return to the
design-commitment approval gate rather than being silently treated as a retry.

```bash
launch_arm dpo 0 1 0
launch_arm entropy 0 0 1
```

- [ ] **Step 4: Evaluate each final adapter on all 400 TEST rows**

Run `eval.py` with the frozen Stage 2A base and each DoRA adapter, TEST hash
`6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618`,
task `comt`, language SDPA, vision FlashAttention2,
`max_sequence_length=32768`, `max_new_tokens=4096`, repetition penalty 1.05,
temperature 0, and top-p 1. Require exactly 400 sample records plus one summary
with finite coverage accuracy, valid-gold accuracy, invalid-gold count, and
inference time.

```bash
evaluate_arm() {
  arm=$1
  run_id="step-pref-dora-$arm-$ablation_stamp"
  mkdir -p "$PWD/results/$run_id/evaluation" "$PWD/logs/$run_id/evaluation"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/transformers/src:$PWD/src" \
    ./venv/ilvr-venv/bin/python eval.py \
    --model_dir "$PWD/checkpoints/$run_id" \
    --base_model_dir "$PWD/checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827" \
    --test_data_path "$PWD/data/TEST.jsonl" --task_name comt \
    --output_json_path "$PWD/results/$run_id/evaluation/predictions.jsonl" \
    --runtime_identity_path "$PWD/results/$run_id/evaluation/evaluation-runtime-identity.json" \
    --max_sequence_length 32768 --max_new_tokens 4096 \
    --repetition_penalty 1.05 --temperature 0 --top_p 1 \
    >"$PWD/logs/$run_id/evaluation/eval.log" 2>&1
}
evaluate_arm ce
evaluate_arm dpo
evaluate_arm entropy
```

- [ ] **Step 5: Publish the comparison and completion evidence**

Inspect all three adapters; validate three evaluation cardinalities and hashes;
authenticate each runtime-identity sidecar and its exact SDPA/FlashAttention2
class names/counts; hash each training-runtime sidecar, predictions JSONL, and
evaluation-runtime sidecar into that run's exact comparison fields; recheck
each `launch-config.json` hash, run ID, triplet, normalized command, and shared
artifact hashes against its training-runtime identity and adapter; require their
Git commit equals the admission identity and tracked-clean is true;
recheck normalized config equality; rank by `correct/400`, preserving exact ties
and using valid-gold accuracy only as secondary reporting. Write canonical
`comparison.json` and a concise Markdown table. State prominently that the
training comparison applies only to the capacity-admitted subset while TEST
evaluation covers all 400 rows.

- [ ] **Step 6: Run final verification once**

Run the full project suite once:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q
```

Then run only distinct artifact/evaluation/config validators needed for the
completion claims: `inspect_stage2_artifact` for all three checkpoints, exact
401-line/evaluation-summary validation and authenticated runtime-identity
sidecar validation for every arm, normalized command
equality, shared hash equality, and strict `comparison.json` schema validation.
Do not repeat an identical broad suite. Record exact commands and complete
outputs before claiming the three experiments finished.
