# Packed Stage 2B Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Admit only authenticated Stage 2B examples whose unchanged natural
packed plan fits the 4608-token capacity envelope, prove the retained boundary
fits DoRA, and finish the CE-only, DPO-only, and entropy-only training and
evaluation runs without OOM.

**Architecture:** Reuse the existing strict Stage 2 length manifest, negative
manifest, production collator, packed planner, one-shot capacity probe, and
launcher. A new strict packed-admission module binds the clean source to one
byte-preserving retained source and source-recomputes every reconstructed
plan. Normal packed Stage 2B training requires the shared admission and verifier
hashes. Single-GPU pre-admission probes use the existing private
`load_production_context` exact-sample path. The distributed capacity CLI is
public but is not an admission bypass: it becomes legal only after the full
admission manifest and verifier authenticate, and it selects one retained
sample from that authenticated cohort in memory.

**Tech Stack:** Python 3.11, PyTorch 2.7.1/CUDA, Qwen2.5-VL, PEFT DoRA,
FlashAttention2 vision attention, SDPA ordinary language attention,
FlexAttention packed decoding, Hugging Face Trainer, Accelerate, DeepSpeed
ZeRO-2, pytest.

**Spec:** `docs/superpowers/specs/2026-09-04-packed-admission-design.md`

**Execution root:** Run every command from
`/root/workspace/ILVR/.worktrees/ours1` on branch `ours1`. The approved design
is commit `0878900`, used only as design provenance. Historical v11/row-tail
evidence was produced at `0e69b54c02f5472c60c6ed64e5eebc2b46374a6c`;
`PilotSummary.git_commit` denotes that historical producer commit, while its
tracked-clean predicate requires both the historical reports' clean flags and a
clean publisher worktree. Tasks 1-6 then create implementation commits. The
single clean HEAD after Task 6 is the **experiment implementation commit**:
Task 7 admission/verifier/capacity/runtime identities and all Task 8 training/
evaluation identities must use it unchanged. No Task 1-6 intermediate commit
is accepted as an experiment identity. Lifecycle progress tracking is
controller state, not an experiment artifact or input to this plan.

## Global Constraints

- Run Python, pytest, training, and evaluation only through
  `./venv/ilvr-venv` with `PYTHONPATH="$PWD/transformers/src:$PWD/src"`.
- Keep model and Trainer state below `checkpoints/`, results below `results/`,
  and every execution log below `logs/`; use one fresh run ID per retry.
- Keep clean `MAX_SEQUENCE_LENGTH=8192` and packed
  `STEP_MAX_PACKED_SEQUENCE_LENGTH=4608` distinct.
- Preserve DoRA rank 32, alpha 64, dropout 0.05; BF16; seed 42; language SDPA;
  vision FlashAttention2; packed FlexAttention; non-reentrant checkpointing;
  and two-rank ZeRO-2.
- Never truncate, split, prune, dynamically skip, or reorder a retained example.
- Require `tracked_worktree_clean=true` when building, verifying, capacity
  validating, launching, and evaluating. Any tracked unstaged or staged change
  fails before model execution; untracked interpreter caches remain ignored.
  Every runtime gate also runs
  `git status --porcelain=v1 --untracked-files=all --ignored=no`: the only
  allowed untracked entries are `??` files whose path contains a `__pycache__`
  component and ends in `.pyc`. Any other untracked path—including Python,
  shell, config, vendored, data, or symlink content—fails before model
  construction. All executable code reached through `PYTHONPATH` must therefore
  be tracked at the experiment implementation commit.
  Before Task 1 publication and before Task 7/8 allocation, require
  `git check-ignore -q --no-index` to accept probe paths under each of
  `results/`, `logs/`, and `checkpoints/`; otherwise stop before writing because
  generated artifacts would self-invalidate the untracked gate.
- The three arms share byte-identical retained source, negatives, admission
  manifest, verifier result, sampler inputs, and all settings except their loss
  coefficient triplet and run/output identities.
- Stop before full negative generation when the frozen v13 premise evidence
  fails authentication. Stop before training when any admission, capacity, or
  ZeRO-2 gate fails. If a
  runtime failure occurs, preserve its fresh run log, make only the causal fix,
  add one causal regression, and retry under a new run ID.
- A fail-closed preflight rejection before model construction is not a runtime
  experiment failure: correct the input/resource conflict and rerun with fresh
  paths if any were allocated. Once model execution begins, any training or
  evaluation failure requires an explicit root-cause record. If the cause is in
  repository code/config/preflight, make the smallest correction plus one
  causal regression; that new commit invalidates Task 7 and every arm. If the
  cause is an external fault for which no honest repository correction/test
  exists (for example power loss, driver reset, or hardware failure), do not
  invent one: preserve evidence, stop the implementing worker, and report a
  blocked run to the root controller/user. The controller decides when the
  external state is restored; no result is claimed in the meantime.
  The worker writes the diagnosis beside the failed execution log as
  `logs/<failed-run-id>/root-cause.txt`, UTF-8 with exactly four LF-terminated
  lines: `phase=<defined phase>`,
  `category=code|config|resource|hardware|external`,
  `evidence=<relative log path>`, and
  `disposition=<correction and causal test, or blocked reason>`. This is a
  concise failure log, not a retry subsystem.
  The one-shot driver maps each subprocess to the exact phase enum, prints the
  failed phase/log path, and exits; it does not speculate about cause. The root
  implementing worker owns diagnosis and writes `root-cause.txt` before any
  correction/retry/design handoff. The same owner wraps each Task 8 shell call
  and applies its explicit arm/evaluation phase.
- The user request is understood as: follow the approved v13 evidence ladder
  and continue with the smallest evidence-supported correction until the fixed
  DoRA method has a no-OOM launch path; do not silently change the scientific
  population or objective.
- Whenever this plan says “return to the design-commitment gate,” the
  implementing worker stops without further runtime mutation and reports the
  observed evidence to the root controller. The root revises/critiques the
  commitment and waits for explicit human approval; no worker invokes or
  assumes that approval itself.
- Capacity admission is an empirical risk gate, not a mathematical proof that
  the four witness axes monotonically upper-bound memory, nor that two steps
  bound allocator behavior over two epochs. Pareto-plus-global witnesses reduce
  the risk of wasting full runs; the actual completion witness is successful
  end-to-end execution of all three arms. The combined-positive probes execute
  the same three raw loss graphs used by every arm, but scalar weighting alone
  does not erase the remaining cohort/order/runtime uncertainty. Report that
  limitation with the results; never call the capacity summary an OOM proof.
  An authentic `oom`, peak fraction above 0.85, distributed capacity failure,
  or 34-or-more packed exclusions is a valid negative scientific result—not a
  retryable defect. Preserve it, stop all later probes/arms, and return the
  evidence to the design-commitment gate for an explicitly approved lower cap
  or population decision. No current commitment authorizes that change.
- Operational phase terms are exact: “runtime evidence starts” when the Task 7
  driver, after all read-only preflights, creates its fresh result/log roots;
  “model construction begins” immediately before the first repository call to
  `from_pretrained`/the equivalent checkpoint loader in a producing process;
  “normal launcher execution” means a packed Stage 2B launch with all capacity-
  only options absent; and evaluation model execution begins immediately before
  `load_evaluation_artifacts`. Preflight tests place sentinels at these calls so
  failure-side-effect assertions use the same boundary.
  Root-cause phase values are exactly
  `premise|clean_build|negative_generation|admission_build|admission_verify|single_capacity|distributed_capacity|capacity_verify|training_ce|training_dpo|training_entropy|evaluation_ce|evaluation_dpo|evaluation_entropy|comparison|final_verify`.

## Wire contracts used by every task

All JSON objects reject duplicate keys and non-finite constants. Canonical JSON
is UTF-8 with sorted object keys, compact separators, `ensure_ascii=False`, and
one final LF; finite floats use Python 3.11 `json.dumps` shortest round-trip
number rendering and booleans are never accepted as numbers. SHA-256 covers
those complete bytes. Stored paths are resolved absolute paths.
every run ID is an ASCII string matching `^[a-z0-9][a-z0-9-]{0,127}$`, with no
slash, dot-segment, whitespace, or control character. Task 7 additionally
requires prefix `step-pref-packed-admission-`; Task 8 arm/comparison IDs require
their exact displayed prefixes. Every derived path is joined below its one
canonical result/log/checkpoint root and checked with `Path.relative_to` after
resolution before any absence check or creation.
Artifact-file readers reject symlinks and non-regular files;
directory identities reject symlinks and require directories that resolve to
the stored absolute path. All readers reject reopen-boundary changes,
unexpected fields, booleans where integers are required, and any hash mismatch.
Symlink rejection applies to the supplied leaf artifact after lexical input
validation; parent symlinks such as the repository's intentional `checkpoints/`
storage link are allowed, then erased by `resolve(strict=True)` and the resolved
absolute target is stored. A supplied leaf symlink is rejected before resolve.
For newly implemented strict file readers, “reopen-boundary” means: `lstat`,
reject symlink/non-regular, open read-only with `O_NOFOLLOW`, require `fstat`
device/inode/size/mtime-ns equals the initial stat, read/hash to EOF, and require
the same `fstat` tuple again before accepting bytes. Validation order is always
path/type, byte read/stability, JSON syntax/duplicates/non-finite, exact schema
and scalar types, supplied hash, cross-artifact identity, reconstructed records,
then runtime predicates; ordered error arrays follow this order and path/source/
rank order within a stage.
Every new public artifact entry point first applies this strict wrapper, then
may delegate decoded semantics to a named existing reader. Existing modules are
not assumed to gain reopen hardening globally and need not be retrofitted for
unrelated callers. Strict readers reconstruct the decoded value's canonical bytes and require
exact byte equality, so alternate whitespace/key order, missing final LF, and
other noncanonical-but-valid JSON are rejected. Fail-artifact validators stop
at the first error in the stated validation order and publish exactly one error
string; they do not accumulate later dependent errors. For verification hashes,
the dependency order is manifest, clean source, retained source, negative,
negative manifest: failure leaves that item and every later dependent item
null, while earlier authenticated hashes remain. Distributed `worker_errors`
is the sole multi-error array and contains at most one nonempty string per rank,
ordered by rank.
Git cleanliness means `git diff --quiet` and
`git diff --cached --quiet`; untracked interpreter caches do not change the
tracked-worktree flag.

`SubsetSourceMap` schema 1 has exact fields
`schema_version,kind,original_source_sha256,subset_sha256,original_source_indices,sample_ids`.
Its kind is `stage2b-subset-source-map`; indices are unique increasing
zero-based original JSONL ordinals, sample IDs are unique nonempty strings in
the same order, and the subset bytes must equal those original lines with one LF
each. Task 7 writes `clean-source-map.json` for all 3354 clean rows. Both
packed-admission build
and verify require `--source-map-path` and its expected
`--source-map-sha256`; this preserves original identity even for a one-row
focused verifier fixture.

`PackedAdmissionIdentity` has exact fields
`original_data_path,original_data_sha256,clean_data_path,clean_data_sha256,source_map_path,source_map_sha256,retained_data_path,retained_data_sha256,clean_length_manifest_path,clean_length_manifest_sha256,image_root_path,checkpoint_path,checkpoint_files,tokenizer_config_identity,negative_path,negative_sha256,negative_manifest_path,negative_manifest_sha256,assistant_plan_rendering,attention_implementation,vision_attention_implementation,latent_size,clean_model_limit,clean_prefilter_limit,max_packed_sequence_length,git_commit,tracked_worktree_clean`.
`checkpoint_files` reuses every `(load_root,relative_path,size,sha256)` entry
from the strict schema-3 negative manifest, sorted by
`(load_root,relative_path)`, and is revalidated from disk. Each checkpoint-file
object has exact fields `relative_path,size,sha256,load_root`, with the types
and path constraints of `step_preference.negatives.CheckpointFile`.
`tokenizer_config_identity` is exactly the existing `_tokenizer_identity`
SHA-256 over the checkpoint inventory's processor/tokenizer entries: an array
of `{relative_path,sha256,size}` objects in inventory order, serialized with
sorted keys and compact separators. The identity builder derives every path by
`Path.resolve(strict=True)`, every file hash from bytes reopened immediately
before publication, `clean_model_limit=8192` as the fixed training model limit,
`clean_prefilter_limit` and rendering/latent values from the authenticated
clean-length manifest (required to be 4608, adjacent-text-space-lf-v1, and 8),
`max_packed_sequence_length` from the build CLI (required to be 4608), language
and vision backends as the fixed packed-Stage-2B values `sdpa` and
`flash_attention_2`, and Git fields from `git rev-parse HEAD` plus
the two clean-diff checks. It cross-checks the original/clean/checkpoint,
tokenizer, negative, and rendering identities against the strict clean and
negative manifests; callers do not supply derived identity values independently.
The build CLI therefore needs no duplicate options for these derived values;
tests reject any manifest/checkpoint/config disagreement with the fixed values.

`PackedAdmissionMeasurement` has the first ten measured fields listed in Task
3, omitting `admission_reason`. `PackedAdmissionRecord` adds the eleventh field
`admission_reason`; only `select_packed_admission` creates classified records,
using `dataclasses.replace`/construction with `retained` or
`packed_length_exceeds_limit`. Reconstruction never invents a reason.
Source index is a nonnegative integer; sample ID is nonempty; clean/unpadded/
padded lengths are positive with `padded >= unpadded >= clean` and padded length
divisible by 128; vision/image counts are nonnegative; reasoning-block, branch,
and scored-token counts are positive. Records are frozen and booleans are
rejected for every integer field.
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
`build_boundary_witnesses(retained_records)` is the sole constructor: it builds
all sorted tie classes, the sorted Pareto representative indices, the global
maximum, and the sorted deduplicated final union, then rejects an empty input or
more than eight final probes. `select_capacity_witness_indices` is a thin return
of that object's `final_probe_source_indices`.
`PackedAdmissionCounts` has exact integer fields
`original,clean_admissible,mapped,retained,filtered`. `original` is always the
line count of `original_data_path`; `clean_admissible` is always the retained
count of the bound clean-length manifest; `mapped` is the number of records in
the supplied source map. The strict invariants are
`mapped == len(records) == retained + filtered` and
`mapped <= clean_admissible <= original`. Therefore the full build records
`3450,3354,3354,...`; focused unit tests also cover a one-row mapped build with
`3450,3354,1,1,0` to prove original source indices are never renumbered.
`PackedAdmissionManifest` has exact fields
`schema_version,kind,identity,records,retained_source_indices,retained_sample_ids,filtered_source_indices,filtered_sample_ids,boundary_witnesses,counts`;
schema is 1 and kind is `stage2b-packed-admission`. `records` contains every
mapped row exactly once in increasing original `source_index` order regardless
of partition; retained/filtered index and sample arrays are separately
increasing partitions of that ordered array.

`PackedAdmissionVerification` has exact fields
`schema_version,kind,status,manifest_path,manifest_sha256,clean_admissible_sha256,packed_retained_sha256,negative_sha256,negative_manifest_sha256,verified_record_count,retained_source_indices,filtered_source_indices,errors`;
kind is `stage2b-packed-admission-verification`, status is `pass|fail`, `errors`
is an ordered array of nonempty strings in detection order, and a failure has at
least one error. Hash fields are `str | null`: each starts null and is populated
only after that artifact's bytes and expected binding authenticate. If the
manifest is unreadable or its supplied hash disagrees, `manifest_sha256` and
all dependent hashes remain null. Any failure leaves record count zero and both
index arrays empty, while already independently authenticated nondependent
hashes may remain populated. A manifest-path read failure still writes
this bounded fail result to the independently supplied output path. Training
accepts only a pass result with no null hash. The verifier resolves no path relative to its
invocation directory: it opens only the absolute paths frozen in the manifest.

`PilotSummary` has exact fields
`schema_version,kind,status,run_id,git_commit,tracked_worktree_clean,candidate_cap,rejected_v11_summary,original_data_path,original_data_sha256,checkpoint_path,checkpoint_files,gpu,audits,admitted_source_indices,ignored_source_indices,capacity_reports,errors`.
`PackedCapacitySummary` has exact fields
`schema_version,kind,status,pilot_summary_path,pilot_summary_sha256,gpu_snapshot_path,gpu_snapshot_sha256,admission_manifest_path,admission_manifest_sha256,verification_path,verification_sha256,command_log_path,command_log_sha256,witness_reports,distributed_witnesses,commands,counts,exclusions,gpu_identities,training_population,errors`.
Both use schema 1; their kinds are respectively `stage2b-packed-pilot` and
`stage2b-packed-capacity`, status is `pass|fail`, and `errors` is an array of
strings that is empty exactly on pass.
`rejected_v11_summary` is an exact `{path,sha256}` object and must identify the immutable
failed v11 pilot summary. Every witness report
also has `source_index,sample_id,packed_padded_length,peak_reserved_fraction,status`.
Each `distributed_witnesses` item has exact fields
`source_index,sample_id,evidence_path,evidence_sha256,checkpoint_path,checkpoint_files,training_runtime_identity_path,training_runtime_identity_sha256,status,rank_peak_reserved_fractions`;
the array is sorted by source index and covers exactly the final probe set.
`checkpoint_files` is a sorted nonempty array of exact
`{relative_path,size,sha256}` objects for the three required adapter files;
rank fractions are two finite floats ordered by rank and at most 0.85.
`commands` is an ordered array of nonempty command token arrays (all tokens are
strings), must byte-for-structure equal the authenticated command log's array,
and is bound by the adjacent absolute path/lowercase SHA-256 fields. `counts`
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
`admitted|ignored_over_cap`. The v13 summary has candidate cap 4608, audits
exactly source rows 1241 and 1314, admitted IDs `[1314]`, ignored IDs `[1241]`,
and capacity reports for both rows. It passes only when row 1241 is padded 4864
with a reserved fraction strictly greater than 0.85, row 1314 is padded 4608
with status `pass` and reserved fraction at most 0.85, and the rejected v11
summary authenticates its row-1372 OOM.
The pilot's original source and checkpoint identity must byte-match the later
full admission identity; `checkpoint_files` uses the same complete
`(load_root,relative_path,size,sha256)` entries.

Each single-device capacity report extends the existing schema-1 probe with
exact top-level fields
`schema_version,status,identity,sample,losses,gradient_norm,memory,gpu,elapsed_seconds,errors`.
Status is `pass|oom|error`; errors is an ordered array of nonempty strings and
is empty exactly on pass. `identity` has exact fields
`checkpoint_path,device,dtype,attention_implementation,vision_attention_implementation,tuning_method,gradient_checkpointing,git_commit,gpu_snapshot_path,gpu_snapshot_sha256`.
`sample` has exact fields
`source_row_index,sample_id,clean_length,step_count,branch_count,packed_unpadded_length,packed_padded_length`;
`losses` has exact nullable-float fields `ce,dpo,entropy,total`; `memory` has
exact integer fields
`device_total_bytes,baseline_allocated_bytes,peak_allocated_bytes,baseline_reserved_bytes,peak_reserved_bytes`
plus finite `peak_reserved_fraction`; and `gpu` has exact fields
`physical_index,name,uuid,total_memory_bytes`. A pass requires all sample fields,
four losses, gradient norm, elapsed time, memory, and GPU fields non-null with
their documented positive/nonnegative constraints, snapshot identity match,
and fraction at most 0.85. OOM/error reports preserve this exact shape with
unobserved scalars null and at least one error. `identity`, `sample`, and
`losses` always remain exact objects (with their unobserved member values null);
`memory` and `gpu` are null until their respective baseline/snapshot succeeds,
then are exact objects whose successfully observed fields are retained. Error
reports never omit or add fields.

`PackedGpuSnapshot` has exact fields `schema_version,kind,gpus`; schema is 1,
kind is `stage2b-packed-gpu-snapshot`, and ordered GPU entries have exact fields
`host_physical_index,visible_cuda_index,name,uuid,total_memory_bytes`. Task 7
sets `CUDA_VISIBLE_DEVICES=0,1`, so entries are exactly host/visible `(0,0)` and
`(1,1)`, both named `NVIDIA H100 NVL` with unique UUIDs. One venv process
captures the snapshot immediately before capacity launches: one
`nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader,nounits` call binds
host indices/names/UUIDs, and `torch.cuda.get_device_properties` supplies the
exact CUDA-visible `total_memory_bytes` used as the denominator by probes. No
rounded nvidia-smi MiB value is used as an equality witness. The process
requires CUDA and nvidia-smi names to agree and compares lowercase canonical
UUID hex after stripping nvidia-smi's `GPU-` prefix from both sides. Every single-device probe
receives the snapshot path/hash and must match visible CUDA 0 to host entry 0,
while distributed ranks 0/1 match entries 0/1 through the same CUDA runtime
properties. The final validator repeats both inventories and requires an exact
match.

`PackedCommandLog` has exact fields `schema_version,kind,commands`; schema is 1,
kind is `stage2b-packed-command-log`, and commands is the ordered array of exact
argv string arrays for Task 7's clean build, negative generation, admission
build/verify, every single-device witness probe, every witness-specific
distributed training (that same training argv produces its capacity evidence).
There is no fictitious separate evidence-production command. The log is
published exclusively before the final validator;
shell text is never reparsed to reconstruct argv. Commands that require
environment settings are actually executed as arrays beginning
`/usr/bin/env`, followed by explicit `KEY=value` tokens and the executable
argv, so the recorded array includes every protocol-owned environment value;
subprocesses otherwise inherit `os.environ` unchanged. That ambient environment
is explicitly outside the command-log authenticity claim; CUDA/GPU/backend,
Git, package-code, and training settings used for acceptance are independently
observed in the defined identities. Do not describe this as complete host-
environment provenance. The validator invocation is
itself preserved in its execution log to avoid a self-hash cycle.
The array contains exactly these subprocesses, in order: clean length-manifest
build; shared negative generation; packed-admission build; packed-admission
verify; every sorted witness's single-GPU probe; then every sorted witness's
two-rank launcher.
GPU snapshot/source-map publication, SHA-256 calculation, schema reloads, and
fresh-path checks are in-process driver operations and therefore are not
misrepresented as commands; their output hashes are bound by the admission or
capacity summary. The final capacity validator is excluded only for the stated
self-hash cycle and its exact argv is captured in
`logs/$admission_id/capacity-validator.log`; the subsequent read-only fresh
summary verifier is likewise excluded and captured in
`capacity-summary-verify.log`. Protocol-owned environment keys are
exactly `CUDA_VISIBLE_DEVICES,PYTHONPATH,TOKENIZERS_PARALLELISM` plus every
uppercase launcher variable shown in Step 6. Their canonical order is CUDA,
PYTHONPATH, tokenizer, then the remaining launcher variables in their displayed
Step 6 order; tests compare the complete arrays.

`PackedDistributedCapacityEvidence` has exact fields
`schema_version,kind,run_id,world_size,process_ranks,global_step,optimizer_step,gradient_accumulation_steps,microbatch_count,sample_id,rank_observations,gpu_snapshot_path,gpu_snapshot_sha256,gpu_identities,checkpoint_path,training_runtime_identity_path,training_runtime_identity_sha256,method,zero_stage,finite_loss,finite_gradient,nonzero_gradient,memory_within_limit,worker_errors`.
Schema is 1, kind is `stage2b-packed-distributed-capacity`, world size is 2, ranks are
`[0,1]`, steps are both 2, gradient accumulation is 2, microbatch count is 4,
method is `dora`, ZeRO stage is 2, and each ordered
rank observation has exact fields
`rank,sample_ids,pre_step_gradient_norms,training_loss,peak_reserved_bytes,total_memory_bytes,peak_reserved_fraction`:
the sample list contains the same sample ID four times, the norm list contains
exactly two finite positive floats captured at optimizer boundaries, loss is
the rank-local finite float returned by `trainer.train()` as `training_loss`,
and peak-reserved fraction is the quotient of the two exact byte
counters and at most 0.85. Each ordered GPU
identity has exact fields `rank,physical_index,name,uuid,total_memory_bytes`.
The four health booleans are derived from all rank observations and are true.
It is published collectively and exclusively after training.
The capacity-only launcher exposes
`--step_packed_capacity_evidence_path` / `STEP_PACKED_CAPACITY_EVIDENCE_PATH`;
it requires `--step_packed_capacity_sample_id` /
`STEP_PACKED_CAPACITY_SAMPLE_ID` to select exactly one already-verified retained
sample after full admission authentication;
it also requires `--step_packed_gpu_snapshot`,
`--step_packed_gpu_snapshot_sha256` and the matching launcher variables
`STEP_PACKED_GPU_SNAPSHOT`, `STEP_PACKED_GPU_SNAPSHOT_SHA256`;
it is absent from normal experiment arms and is legal only for packed Stage 2B
with world size 2, ZeRO-2, `max_steps=2`, batch 1, accumulation 2, and exactly
one capacity sample selected from the verified full admission source.

Every final training artifact contains canonical
`stage2_training_runtime_identity.json` with exact fields
`schema_version,kind,run_id,git_commit,tracked_worktree_clean,coefficient_triplet,tuning_method,seed,bf16,per_device_train_batch_size,gradient_accumulation_steps,gradient_checkpointing,use_reentrant,global_step,optimizer_step,objective_metric_name,objective_metric_observation_count,finite_objective_metrics,positive_pre_step_gradient_count,nonfinite_pre_step_gradient_count,first_positive_pre_step_gradient_norms,training_length_manifest_path,training_length_manifest_sha256,admission_manifest_sha256,verification_sha256,world_size,process_ranks,rank_observations`. Schema is 1,
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
identity. Immutable runtime values are exactly `tuning_method="dora"`, seed 42,
BF16 true, per-device batch 1, accumulation 2, checkpointing true, and
`use_reentrant=false`; the publisher reads them from the bound parsed arguments
and constructed `TrainingArguments`, not from shell intent.
The length-manifest path/hash are the actual launcher-generated schema-2
artifact reopened after Trainer completion, never the dry-run zero hash.
Task 8 requires all three manifest byte hashes identical and each manifest to
bind the same retained data, checkpoint processor, latent/rendering/boundary
settings, and 8192 model limit; this directly authenticates equal sampler input.
For normal arms, global/optimizer steps are equal positive integers;
`objective_metric_name` is exactly `loss/ce|loss/dpo|loss/entropy` according to
the sole nonzero coefficient, its observation count is positive, and
`finite_objective_metrics=true`. The existing ZeRO-2 pre-step adapter counts finite
positive reduced/unscaled gradient norms at true optimizer boundaries and
stores only the first two values; positive count must be positive and
`nonfinite_pre_step_gradient_count=0`. Because exactly one coefficient is nonzero, this is the
objective-specific witness that the arm executed a nonzero update path. Capacity
runs use metric `loss/total` and retain their stronger exact two-norm/two-step
contract.

“Launch config” below names one schema; its three concrete filenames are only
`launch-config-ce.json`, `launch-config-dpo.json`, and
`launch-config-entropy.json` in the comparison directory. Each has exact fields
`schema_version,arm,run_id,coefficient_triplet,normalized_command,capacity_summary_path,capacity_summary_sha256,retained_data_sha256,negative_sha256,negative_manifest_sha256,admission_manifest_sha256,verification_sha256`.
Its schema version is 1 and arm is exactly `ce|dpo|entropy`.
The final `comparison.json` has exact fields
`schema_version,kind,primary_metric,training_population,runs`; each run has
`rank,arm,run_id,launch_config_sha256,checkpoint_sha256,training_runtime_identity_sha256,predictions_sha256,evaluation_runtime_identity_sha256,coefficient_triplet,correct,source_count,coverage_accuracy,valid_gold_accuracy,invalid_gold_count,average_inference_time_sec`.
`checkpoint_sha256` is the exact object mapping
`adapter_config.json,adapter_model.safetensors,stage2_peft_metadata.json` to
lowercase SHA-256. Those are the only model-bearing files accepted by this
comparison mapping; processor/tokenizer/config files permitted and authenticated
by `inspect_stage2_artifact` may coexist and are not treated as unexpected model
weights. Any other model/optimizer/scheduler/RNG/shard file in the final root is
rejected. Commands are JSON arrays of strings and coefficient triplets
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
For Task 8's `comt` evaluation, a successful sample row has exactly the fields
currently emitted by `eval.py`:
`index,task_name,image_input,prompt_identity,prompt_token_length,generated_suffix_token_ids,raw_output,assistant_only,prediction,extracted_final_answer,prediction_parse_status,prediction_parse_detail,prediction_normalized,gold_final_answer,gold_parse_status,gold_parse_detail,gold_normalized,invalid_gold_reason,latent_start_count,latent_block_count,stop_reason,match,inference_time_sec`.
An inference-error row has exactly
`index,task_name,error,inference_time_sec,prompt_identity,prompt_token_length,generated_suffix_token_ids,raw_output,assistant_only,prediction,extracted_final_answer,prediction_parse_status,gold_final_answer,gold_parse_status,gold_parse_detail,gold_normalized,invalid_gold_reason,latent_start_count,latent_block_count,stop_reason,match`.
Indices are unique `0..399` in order and `task_name="comt"`; arrays, nullable
fields, strings, booleans, and finite timing/count values retain their concrete
`eval.py` types. On success, index/prompt/count fields are nonnegative integers,
`image_input` is `str|list[str]`, prompt identity is nonempty string,
`generated_suffix_token_ids` is `list[int]`, raw/assistant/prediction/extracted/
normalized/status/stop fields are strings (possibly empty only where parsing
permits), latent counts are nonnegative integers, `match` is boolean, and timing
is finite nonnegative. `gold_final_answer` is `str|null`; prediction/gold parse
detail and invalid-gold reason are the only nullable derived fields. On error,
`error` is nonempty; timing, prompt identity/length, raw/assistant/prediction/
extracted output, latent counts, and stop reason are null; generated token IDs
is empty; match is false; and gold final/parse/detail/normalized/reason retain
their typed parse result. The final summary is line 401, after exactly 400
sample rows, and has exactly
`summary,num_samples,average_inference_time_sec,total_inference_time_sec,accuracy,success,correct,source_count,valid_gold_count,invalid_gold_count,invalid_gold,coverage_accuracy,valid_gold_accuracy`.
It requires `summary=true`, `num_samples=source_count=400`, `success=correct`,
`valid_gold_count+invalid_gold_count=400`, and finite nonnegative timing and
accuracy values. `invalid_gold` is ordered and contains exact
`{index:int,reason:nonempty str}` objects. Metrics are recomputed through
`EvaluationScoreSummary.from_rows`; if `valid_gold_count=0`,
`valid_gold_accuracy` is exactly `0.0`, while coverage is always
`correct/source_count` (also exposed as `accuracy`).
The error-row schema exists so a failed evaluation can be diagnosed without
losing partial output; it is never accepted for an arm comparison.
`validate-arm --phase evaluation` requires zero rows containing `error` or
`prediction_parse_status="inference_error"`. Non-exception prediction statuses
`parsed|empty_prediction|malformed` remain ordinary scored predictions;
prediction detail is nonnull only for malformed. Gold status is
`parsed|invalid_gold|malformed`; invalid/malformed gold carries a nonempty
reason and never matches.
These schemas are implemented and unit-tested in the task that first writes
them when they are durable code outputs. Task 1 uses the tested premise
publisher; Task 8 uses the tested ablation reporter implemented before runtime
materialization. One-off ad-hoc JSON shapes are not accepted.

Reused repository contracts are pinned to these source symbols and their tests,
not guessed from comments: schema-2 clean manifests use
`src/stage2/length_manifest.py::{Stage2LengthManifest,read_stage2_length_manifest}`;
schema-3 negatives/checkpoint inventory use
`src/step_preference/negatives.py::{NegativeManifest,CheckpointFile,inspect_stage2a_checkpoint_identity,load_negative_store}`;
sample IDs use `src/step_preference/spans.py::canonical_sample_id`; plan geometry
uses `src/step_preference/packing.py::build_packed_tree_plan`; checkpoint kind
uses `src/stage2/checkpointing.py::inspect_stage2_artifact`; and backend class
validation uses
`src/stage2/attention_backend.py::validate_qwen2_5_vl_attention_backends`.
Tasks modifying consumers add focused compatibility tests against these exact
symbols; no duplicated alternative interpretation is allowed.

The model's clean tokenizer limit remains 8192. The clean 4608 pass in Task 7
is a lossless prefilter for a packed cap of 4608 because producer and verifier
both assert `packed_unpadded_length >= clean_length`; it does not change the
model or evaluation limit.

---

### Task 1: Authenticate and normalize the observed v13 capacity-tail evidence

**Files:**

- Read: `data/TRAIN.jsonl`
- Read: `checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827/`
- Read: `results/step-pref-packed-pilot-20260904T183536Z/pilot-summary.json`
- Read: `results/step-pref-packed-cap-search-20260905T060232Z-row1241/`
- Read: `results/step-pref-packed-cap-search-20260905T062915Z-row1314/`
- Create: `scripts/publish_packed_v13_premise.py`
- Create: `tests/step_preference/test_packed_v13_premise.py`
- Write: `results/step-pref-packed-premise-v13-<UTC>/pilot-summary.json`
- Write: `logs/step-pref-packed-premise-v13-<UTC>/validate.log`

**Interfaces:**

- Consumes: immutable failed-v11 summary plus the authenticated row-1241 and
  row-1314 audits, manifests, capacity reports, and current frozen inputs.
- Produces: one exact `PilotSummary` that records candidate cap 4608, the
  immediate rejected upper boundary, the accepted 24-branch boundary, and the
  rejected-v11 premise chain without repeating GPU work.
- Produces CLI `scripts/publish_packed_v13_premise.py check|publish`. Both
  subcommands take required `--original-data`, `--checkpoint`,
  `--rejected-v11-summary`, `--row1241-root`, and `--row1314-root`; `publish`
  additionally requires one fresh `--output-path`. `check` performs every
  authentication and live-GPU check but writes nothing. `publish` repeats the
  checks, exclusively writes the exact summary, reloads it through the same
  strict schema decoder, and prints its path/hash. No runtime training occurs.
- Historical v11 input is one exact object with fields
  `schema_version,run_id,git_commit,tracked_worktree_clean,original_train,checkpoint,gpus,probe_gpu,audits,capacity_reports,admitted_row_ids,ignored_over_cap_row_ids,unprobed_admitted_row_ids,failure,validation,overall_status`.
  `original_train={path,sha256}`; checkpoint has
  `path,checkpoint_files,identity_source_manifests`; GPU and probe-GPU objects
  are respectively `{index,name,uuid,memory_total_mib,memory_used_mib}` and
  `{host_gpu_index,visible_device_index,name,uuid}`; identity-source refs are
  `{row_id,path,sha256,schema_version}`; audit refs are
  `{row_id,path,sha256,padded_length,status}`; capacity refs are
  `{row_id,path,sha256,status,device_total_bytes,peak_reserved_bytes,reserved_ratio}`;
  failure is `{row_id,observed_status,reason,errors}` and validation contains
  exactly the boolean keys
  `all_admitted_reports_pass,all_admitted_reports_present,at_least_two_admitted,reserved_ratio_at_most_0_85,row_1582_admitted`.
  Lists preserve source/GPU order.
  This publisher accepts only the frozen observations needed here:
  `overall_status="fail"`, audit status `"admitted"`, capacity and observed
  status `"oom"`, failure reason `"capacity_probe_oom"`, and row 1372; it is
  not a general decoder for hypothetical historical statuses.
  Each historical packed audit has exact fields `schema_version,identity,records,winner`;
  identity is
  `{checkpoint_path,data_path,image_root,negative_path,negative_manifest,negative_sha256,max_sequence_length,assistant_plan_rendering,attention_implementation,vision_attention_implementation}`
  and record/winner is
  `{source_index,sample_id,clean_length,step_count,branch_count,unpadded_length,padded_length}`.
  Each historical capacity report has exact fields
  `schema_version,status,identity,sample,losses,gradient_norm,memory,elapsed_seconds,errors`,
  with identity/sample/loss/memory keys matching the single-device schema above
  except the historical report has no GPU/snapshot fields. The strict schema-3
  negative manifest and checkpoint file object are exactly the already-defined
  `NegativeManifest`/`CheckpointFile` contracts. Tests pin all field sets and
  allowed v11 status/failure values; no unknown historical shape is accepted.

- [ ] **Step 1: Write failing premise decoder/publisher tests**

Use small immutable fixtures for the exact v11, audit, capacity-report,
schema-3 negative-manifest, checkpoint-inventory, and live-GPU interfaces.
Inject only the live-GPU inventory reader. Test the accepted 4608 and rejected
4864 inequalities, exact historical hashes/commit, complete matching checkpoint
inventory, exclusive output, duplicate-key/non-finite/unknown-field rejection,
stale GPU identity, and a mutated v11/row report/manifest. Assert `check` writes
nothing and `publish` produces the exact `PilotSummary`.

- [ ] **Step 2: Run RED, implement the minimal strict publisher, and commit**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_packed_v13_premise.py
```

Expected RED: import or missing-CLI failure. Implement only the two subcommands
and exact schema described above, reuse `CheckpointFile` validation and the
repository canonical JSON helpers, then rerun the same test and commit:

```bash
git add scripts/publish_packed_v13_premise.py \
  tests/step_preference/test_packed_v13_premise.py
git commit -m "test: authenticate packed v13 premise"
```

- [ ] **Step 3: Authenticate immutable inputs before allocating an output ID**

```bash
test -z "$(git diff --name-only)"
test -z "$(git diff --cached --name-only)"
git check-ignore -q --no-index results/.codex-ignore-probe
git check-ignore -q --no-index logs/.codex-ignore-probe
git check-ignore -q --no-index checkpoints/.codex-ignore-probe
test "$(sha256sum data/TRAIN.jsonl | cut -d' ' -f1)" = 5bc1313a594f5713975b67388c745c6632275c37fd260055b27230c9cfedbed8
test "$(sha256sum results/step-pref-packed-pilot-20260904T183536Z/pilot-summary.json | cut -d' ' -f1)" = 7419aff8604072b57966fb4b460f99735cb6f7f2c2b386d2f46d7a28139a2c6f
test "$(sha256sum results/step-pref-packed-cap-search-20260905T060232Z-row1241/row-1241-capacity.json | cut -d' ' -f1)" = 82a0117a65b997ef66bfa49d8762bc443872a72ae68cb4d921fd1abf6a4430ca
test "$(sha256sum results/step-pref-packed-cap-search-20260905T060232Z-row1241/row-1241-packed-audit.json | cut -d' ' -f1)" = 5e3cd459a72c4f9a29f61a8c9f3b5c2ef06ae351f8bb7f4024f929ab342d6352
test "$(sha256sum results/step-pref-packed-cap-search-20260905T060232Z-row1241/row-1241-negative-manifest.json | cut -d' ' -f1)" = febc23b80a9953de643cd7c972e71a4beac1de5d79fad9bfc47c59a775f052d5
test "$(sha256sum results/step-pref-packed-cap-search-20260905T062915Z-row1314/row-1314-capacity.json | cut -d' ' -f1)" = 8785cd4283c95d0bac6c5e96531342a0b565b355cf9805037d74bc2a6ce82fff
test "$(sha256sum results/step-pref-packed-cap-search-20260905T062915Z-row1314/row-1314-packed-audit.json | cut -d' ' -f1)" = c4cbac9305439f7de50cc7503fb26e6639e2266208a16ed17725649e06a014a0
test "$(sha256sum results/step-pref-packed-cap-search-20260905T062915Z-row1314/row-1314-negative-manifest.json | cut -d' ' -f1)" = aa2f591ca323cd99af084d38ebbffb30e46187294bfd2ac760cda03ccb76d8c4
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  scripts/publish_packed_v13_premise.py check \
  --original-data data/TRAIN.jsonl \
  --checkpoint checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
  --rejected-v11-summary results/step-pref-packed-pilot-20260904T183536Z/pilot-summary.json \
  --row1241-root results/step-pref-packed-cap-search-20260905T060232Z-row1241 \
  --row1314-root results/step-pref-packed-cap-search-20260905T062915Z-row1314
```

The CLI requires the v11 summary status `fail` and its row-1372 OOM, row 1241
`(steps,branches,clean,padded)=(6,24,4000,4864)` with exact fraction
`85115011072/99949674496 > 0.85`, row 1314 `(6,24,3733,4608)` with exact
fraction `78922121216/99949674496 <= 0.85`, four finite losses and one finite
positive gradient in each capacity report, identical complete checkpoint
inventories, producing commit
`0e69b54c02f5472c60c6ed64e5eebc2b46374a6c`, `cuda:0`, and the current live
GPU-0 identity. Expected: `VERIFIED_V13_BOUNDARY rejected=4864 accepted=4608`;
no output directory exists yet.
“Current live GPU 0” means the same host physical index 0 with the exact
historical UUID, name, and CUDA total-memory bytes—not merely another H100 of
the same model. If that UUID is unavailable, Task 1 fails and the worker reports
the hardware-evidence blocker; substitution requires a new approved premise.

- [ ] **Step 4: Allocate once, publish, and strictly reload the summary**

```bash
premise_id="step-pref-packed-premise-v13-$(date -u +%Y%m%d-%H%M%S)"
test ! -e "results/$premise_id"
test ! -e "logs/$premise_id"
mkdir -p "results/$premise_id" "logs/$premise_id"
set -o pipefail
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  scripts/publish_packed_v13_premise.py publish \
  --original-data data/TRAIN.jsonl \
  --checkpoint checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
  --rejected-v11-summary results/step-pref-packed-pilot-20260904T183536Z/pilot-summary.json \
  --row1241-root results/step-pref-packed-cap-search-20260905T060232Z-row1241 \
  --row1314-root results/step-pref-packed-cap-search-20260905T062915Z-row1314 \
  --output-path "results/$premise_id/pilot-summary.json" \
  2>&1 | tee "logs/$premise_id/validate.log"
```

Expected: `VERIFIED_V13_PREMISE status=pass cap=4608`, exact schema, empty
errors, and no rewrite of historical inputs. Any mismatch stops before Task 2.
The publisher's final line is exactly
`PILOT_SUMMARY_PATH=<absolute path> PILOT_SUMMARY_SHA256=<digest>`; the root
controller records that output and supplies the absolute path as
`PILOT_SUMMARY_PATH` in Task 7. No shell inheritance is assumed.

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
- Launcher owns the matching `STEP_MAX_PACKED_SEQUENCE_LENGTH` (default 4608),
  `STEP_PACKED_ADMISSION_MANIFEST`,
  `STEP_PACKED_ADMISSION_MANIFEST_SHA256`,
  `STEP_PACKED_ADMISSION_VERIFICATION`, and
  `STEP_PACKED_ADMISSION_VERIFICATION_SHA256` variables.
- Launcher also owns `SEED` (default 42) and forwards `--seed`. Optional
  `STAGE2_DRY_RUN_FORMAT=json` is legal only with `STAGE2_DRY_RUN=true` and
  emits exactly one compact JSON array containing the final process argv, with
  no environment prefix or shell quoting; the default dry-run text remains
  backward compatible. Dry run performs argument/config preflights but creates
  no checkpoint/result/log directory. It always synthesizes exactly one
  `--stage2_length_manifest <repo>/results/<run-id>/stage2-length-manifest.json`
  plus one `--stage2_length_manifest_sha256 <64 zeros>` pair. In JSON dry-run,
  the only run-derived value tokens are those following `--save_model_path`,
  `--log_file`, and `--stage2_length_manifest`; there is no diagnostic stdout.

- [ ] **Step 1: Write failing immutable-config and launcher tests**

Add tests that assert:

```python
args = parse_args(required_stage2b_packed_args + [
    "--step_max_packed_sequence_length", "4608",
    "--step_packed_admission_manifest", "admission.json",
    "--step_packed_admission_manifest_sha256", "a" * 64,
    "--step_packed_admission_verification", "verification.json",
    "--step_packed_admission_verification_sha256", "b" * 64,
])
assert args.step_preference_config.max_packed_sequence_length == 4608
assert args.step_preference_config.packed_admission_manifest == "admission.json"
```

Parametrize zero/negative cap rejection only when packing is enabled, partial
path/hash pairs, malformed SHA-256, and launcher values `0`, empty hash, or one
missing admission field. Assert ordinary and non-packed Stage 2 command tokens
remain unchanged except for the newly explicit, semantics-preserving
`--seed 42` pair. Assert the launcher forwards all five options and treats them
as managed. Assert seed 42 is explicit. Assert JSON dry-run round-trips tokens
containing spaces without `shlex`, writes no artifact directory, and rejects an
unknown format or JSON format without dry-run.
Assert the exact synthesized length-manifest/hash pair and that no other token
contains either test run ID.

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
4608 at the public parser and must be a non-boolean positive integer whenever
`packing_enabled` is true. Do not require admission artifacts in the dataclass;
Task 5 enforces them only for normal packed Stage 2B training so the existing
private capacity loader can reconstruct pre-admission rows.

- [ ] **Step 4: Implement launcher defaults and fail-fast forwarding**

When `STEP_PREFERENCE_PACKING=true`, require all four admission artifact
variables for normal launcher execution, validate both hashes with
`^[0-9a-f]{64}$`, and append all five CLI options. Add them to
`MANAGED_OPTIONS`. Keep `--stage2b_preflight_only` usable with these same
artifacts. Do not create output/result/log directories before the preflight
subprocess succeeds. Implement the seed and JSON dry-run contract above without
changing the existing default dry-run format.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_config.py tests/stage2/test_launcher.py
git add src/step_preference/config.py src/utils_deepseed.py \
  run_stage2_training.sh tests/step_preference/test_config.py \
  tests/stage2/test_launcher.py
git commit -m "feat: configure packed stage2b admission"
```

Expected: both focused test modules pass.

---

### Task 3: Implement strict packed-admission records and witness selection

**Files:**

- Create: `src/step_preference/packed_admission.py`
- Create: `tests/step_preference/test_packed_admission.py`

**Interfaces:**

- Produces frozen dataclasses `PackedAdmissionIdentity`,
  `PackedAdmissionMeasurement`, `PackedAdmissionRecord`, `PackedTieClass`,
  `PackedBoundaryWitnesses`,
  `PackedAdmissionCounts`, `PackedAdmissionManifest`, and
  `PackedAdmissionVerification`.
- Produces `select_packed_admission(measurements, *, max_packed_length: int) ->
  tuple[tuple[PackedAdmissionRecord, ...], tuple[PackedAdmissionRecord, ...]]`.
- Produces `build_boundary_witnesses(records) -> PackedBoundaryWitnesses`.
- Produces `select_capacity_witness_indices(records) -> tuple[int, ...]` using
  the Pareto-plus-global-maximum union from v13.
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
winner = record(4, padded=4608, unpadded=4590, clean=3733,
                vision=100, branches=8, scored=400)
dominator = record(3, padded=4608, unpadded=4550, clean=3600,
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
class PackedAdmissionMeasurement:
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

@dataclass(frozen=True)
class PackedAdmissionRecord(PackedAdmissionMeasurement):
    admission_reason: str

def select_packed_admission(measurements, *, max_packed_length):
    retained = tuple(
        PackedAdmissionRecord(**asdict(row), admission_reason="retained")
        for row in measurements
        if row.packed_padded_length <= max_packed_length
    )
    filtered = tuple(
        PackedAdmissionRecord(
            **asdict(row), admission_reason="packed_length_exceeds_limit"
        )
        for row in measurements
        if row.packed_padded_length > max_packed_length
    )
    return retained, filtered

def select_capacity_witness_indices(retained_records):
    return build_boundary_witnesses(
        retained_records
    ).final_probe_source_indices

def build_boundary_witnesses(retained_records):
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
    return PackedBoundaryWitnesses(
        tie_classes=build_sorted_tie_classes(representatives, records),
        pareto_source_indices=tuple(sorted(row.source_index for row in pareto)),
        global_max_source_index=maximum.source_index,
        final_probe_source_indices=tuple(sorted(selected)),
    )
```

`build_sorted_tie_classes` emits only tuples with at least two members and uses
the exact ordering/representative contract above. All constructors validate
their exact nested types, ordering, uniqueness, allowed strings, and nonempty
witness rules; the illustrative code omits only those already-specified checks.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_packed_admission.py
git add src/step_preference/packed_admission.py \
  tests/step_preference/test_packed_admission.py
git commit -m "feat: define packed admission manifest"
```

---

### Task 4: Build and source-recompute the shared admission artifacts

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
- Producer and verifier both rebuild plans through the shared
  `Stage2DataCollator` and `build_packed_tree_plan`; verifier never calls the
  producer or trusts its record/partition values as expected input. This is an
  independent recomputation from authenticated source bytes and configuration,
  not an implementation-independent second planner.

- [ ] **Step 1: Write failing record-feature and build tests**

Mock only checkpoint/processor loading. Assert feature definitions:

```python
vision_token_count = int((batch["input_ids"] == model_config.image_token_id).sum())
image_count = len(prepared.original_image_paths)
reasoning_block_count = len(batch["reasoning_step_spans"])
branch_count = len(plan.branches)
scored_token_count = int(plan.selected_positions().numel())
```

Assert `build` validates the original schema-2 manifest at clean cap 4608,
requires every source-map index to belong to that manifest's retained indices,
byte-reconstructs the supplied clean JSONL from exactly the mapped original
indices, validates
schema-3 negatives against the clean source, writes the retained JSONL in
original order, records all 3354 clean-admissible rows, and binds every input
hash plus git commit and required `tracked_worktree_clean=true`. A dirty tracked
tree makes both build and verify fail without publishing a pass artifact. Add a
one-row source-map fixture whose
original index is 1314 and assert both its admission record and retained index
remain 1314 rather than being renumbered to zero.

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
    measurements = reconstruct_natural_packed_records(
        checkpoint_path=args.checkpoint_path,
        data_path=args.clean_data_path,
        image_root=args.image_root,
        negative_path=args.negative_path,
        negative_manifest=args.negative_manifest,
        original_source_indices=clean.original_source_indices,
        max_packed_length=args.max_packed_sequence_length,
    )
    retained, filtered = select_packed_admission(
        measurements, max_packed_length=args.max_packed_sequence_length
    )
    witnesses = build_boundary_witnesses(retained)
    manifest = assemble_manifest(
        args, clean,
        tuple(sorted((*retained, *filtered), key=lambda row: row.source_index)),
        retained, filtered, witnesses,
    )
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

Build preflights that both output paths are absent before reconstruction. It
writes retained bytes first and the manifest second because the manifest hashes
the retained file. If the second exclusive write fails, the run is failed and
the immutable orphan retained file is preserved as failure evidence; retry uses
a fresh run ID. Do not add rollback or overwrite machinery.

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
- Modify: `scripts/generate_step_negatives.py`
- Create: `scripts/run_packed_admission_protocol.py`
- Create: `scripts/validate_packed_capacity_gate.py`
- Create: `scripts/publish_dora_ablation.py`
- Modify: `tests/step_preference/test_packed_capacity_probe.py`
- Modify: `tests/step_preference/test_negative_generation.py`
- Create: `tests/step_preference/test_packed_capacity_evidence.py`
- Create: `tests/step_preference/test_packed_capacity_gate.py`
- Create: `tests/step_preference/test_packed_admission_protocol.py`
- Create: `tests/step_preference/test_dora_ablation_reporting.py`
- Modify: `tests/step_preference/test_stage2b_trainer.py`
- Modify: `tests/stage2/test_launcher.py`
- Modify: `tests/stage2/test_eval_integration.py`
- Modify: `tests/stage2/test_attention_backend.py`

**Interfaces:**

- Capacity JSON records GPU name/UUID, physical index, and
  `peak_reserved_fraction`; status is pass only when the fraction is at most the
  CLI `--max_peak_reserved_fraction` default 0.85.
- Negative generation owns `--seed` default 42, seeds Python/Torch/CUDA before
  model loading, records `seed=42` and `decoding="greedy-argmax-v1"` in schema-3
  generation settings, and keeps sampling disabled. The algorithm chooses
  argmax tokens only; the seed freezes incidental runtime RNG rather than
  defining a stochastic objective.
- The probe CLI accepts optional `--sample_id`; when present it passes that ID
  to `load_production_context` and requires the resulting batch/report identity
  to match it.
- The probe also requires `--gpu-snapshot` and `--gpu-snapshot-sha256`; the
  capacity-only launcher owns matching
  `STEP_PACKED_GPU_SNAPSHOT` / `STEP_PACKED_GPU_SNAPSHOT_SHA256` and forwards
  them with the distributed evidence path. It also owns required
  `STEP_PACKED_CAPACITY_SAMPLE_ID` and filters the already-authenticated full
  retained dataset to that exact sample only on the capacity-evidence path.
- `utils_deepseed.parse_args` exposes optional
  `args.step_packed_capacity_sample_id`; launcher forwarding requires it iff
  `STEP_PACKED_CAPACITY_EVIDENCE_PATH` is set, and `src.main.run_stage2`
  performs the exact retained-sample selection after admission validation.
- `validate_packed_capacity_gate.py publish` consumes one verified admission manifest,
  one capacity JSON and one distributed training evidence/checkpoint pair per
  final witness; it writes one exclusive `capacity-summary.json` and exits zero
  only if all exact identities and gates pass.
- The validator CLI requires `--admission-manifest`,
  `--admission-manifest-sha256`, `--pilot-summary`,
  `--pilot-summary-sha256`, `--gpu-snapshot`, `--gpu-snapshot-sha256`,
  `--commands-path`, `--commands-sha256`,
  `--verification`, `--verification-sha256`, one or more repeated
  `--capacity-report`, repeated `--distributed-evidence`, repeated
  `--distributed-checkpoint`, and `--output-path`. The two distributed arrays
  must have the same length and correspond by source-index identity, not merely
  CLI position.
  Its `verify` subcommand requires only `--summary`, `--summary-sha256`, and the
  live GPU snapshot path/hash; it strictly reopens every path/hash transitively
  bound by the summary and prints the v13 verification line without writing.
- `run_packed_admission_protocol.py` is a direct one-shot Task 7 driver, not a
  retry/recovery framework. It accepts the authenticated pilot path/hash plus
  `--run-id`, executes the fixed clean-build, negative, admission, probe,
  distributed-witness, and final-gate argv arrays sequentially via
  `subprocess.run(..., check=True)`, and owns the in-memory ordered command list
  for the entire process. It writes each subprocess stdout/stderr to the exact
  log path specified in Task 7, stops on the first failure, and exclusively
  publishes the canonical command log only after the last producing command.
  The final validator invocation is logged but excluded from the self-hashed
  command list. The driver has no retry, resume, cleanup, or overwrite path.
- `publish_dora_ablation.py freeze-launches` requires
  `--comparison-dir`, `--capacity-summary`, its SHA-256, and exact
  `--{ce,dpo,entropy}-run-id`/`--{ce,dpo,entropy}-command-json` pairs. It
  strictly parses the three launcher JSON dry-run strings, normalizes only the
  six specified option values, creates the fresh comparison directory, and
  exclusively writes the three exact launch configs. `publish-comparison`
  requires that comparison directory, the same capacity summary/hash, and
  `--test-data`/its frozen SHA-256; it derives each standardized checkpoint,
  prediction, and runtime-identity path from the authenticated run IDs, strictly
  authenticates all three arms, and exclusively writes the exact comparison
  JSON and Markdown table. It has no training/evaluation responsibility and
  refuses partial/duplicate arms. `validate-arm --phase training|evaluation`
  derives one arm's paths and performs the corresponding non-writing gate;
  evaluation phase additionally requires `--test-data` and its SHA-256.
  `verify-comparison` strictly reopens the three configs, comparison JSON,
  Markdown, capacity summary, TEST data, and all derived arm artifacts without
  writing. `preflight --phase freeze|training|evaluation` is silent on success,
  authenticates the capacity/GPU snapshot plus implementation Git/executable-
  cleanliness gates, and requires `--arm` for training/evaluation. All five
  subcommands use the same decoder/validator functions.
- The new distributed evidence publisher is deliberately separate from the
  existing four-rank benchmark evidence protocol. Each launch gathers exact sample IDs
  observed by both ranks, authenticates the final output-root artifact saved
  after step 2, and writes one `PackedDistributedCapacityEvidence` on rank zero
  after all ranks finish.
- Every capacity GPU must have name exactly `NVIDIA H100 NVL`. Immediately
  before launch, bind each visible physical index to its live `nvidia-smi` UUID;
  the report/evidence name, UUID, index, and total-memory bytes must match that
  live snapshot. Apply this predicate to every single-device witness and both
  distributed ranks for every witness, not only the premise probe.

- [ ] **Step 1: Write failing headroom and identity tests**

Extend the fake runtime with `device_name()` and `device_uuid()`. Test exact
ratios 0.85 (pass) and greater than 0.85 (error), non-H100 identity, non-finite
loss, zero gradient, mismatched sample/length, duplicate evidence paths, and an
existing output. Test that the final witness set equals the manifest's complete
Pareto-plus-maximum set rather than a caller-selected subset.
Extend negative-generation tests to require explicit seed 42, greedy decoding,
no sampling option, deterministic generation settings, and consumer rejection
when either seed or decoding identity is mutated.

Add failing publisher/launcher tests for the exact two-rank schema: each rank
must observe the sole admitted sample exactly four times; `global_step` and
`optimizer_step` must both be 2, gradient accumulation must be 2, and
microbatch count must be 4; loss and gradient must be finite/nonzero; every
rank's exact peak-reserved fraction must be at most 0.85; all worker error
arrays must be empty; and only rank zero may publish after the collective
gather. Reject world sizes other than two, any different sample,
partial/malformed evidence path configuration, a non-ZeRO-2 run, or a missing
final output-root Stage 2 DoRA artifact. Reject a missing, duplicate, unknown,
or non-retained capacity sample ID before model construction; normal arms reject
the private capacity-sample setting entirely.
Test a swapped rank/visible index, stale UUID, wrong total memory, changed live
snapshot, and missing snapshot hash; each fails before model execution or final
capacity publication.
Also mutate the premise-summary status/cap, original-source hash, rejected-v11
chain, or any checkpoint-file hash and assert the final gate rejects it against
the admission identity.

In `test_packed_admission_protocol.py`, stub `subprocess.run` and GPU inventory
only. Assert the driver constructs the exact ordered argv arrays, passes the
same full shared artifacts to every witness, uses each manifest source/sample
pair exactly once, writes the canonical command log from the arrays rather than
  shell text, stops without starting a later command after one failure, and
  rejects existing run result/log targets before its first state-changing
  subprocess and each derived witness checkpoint/result/log target immediately
  before that witness launch. Assert environment-dependent launches are executed
  through their recorded `/usr/bin/env KEY=value ...` arrays rather than an
  unrecorded `env=` mapping.

In `test_dora_ablation_reporting.py`, cover exact launch/comparison schemas,
all three triplets and run-ID/path normalization, malformed or noisy dry-run
JSON, an unnormalized run ID, shared-token drift, existing comparison outputs,
adapter file hashes, runtime identities, exact 400+summary evaluation schemas,
zero-valid-gold metrics, competition ranking/ties, and deterministic Markdown
at `comparison.md`. Exercise non-writing training/evaluation arm validation and
post-publication `verify-comparison`, including one mutation per bound artifact.
Test silent preflight ownership for dirty Git, unexpected untracked files, busy
or changed GPUs, wrong commit, missing/extra arm, and each legal phase.

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
also require the exact DoRA/seed/BF16/batch/accumulation/checkpointing/non-reentrant
fields, and make any one-field mutation fail capacity/arm validation. Ordinary
or non-packed runs may omit this sidecar.
For each of the three one-hot coefficient triplets, simulate at least one true
optimizer boundary and assert the sidecar selects the matching raw metric,
records positive global/optimizer steps, finite objective metrics, a positive
pre-step-gradient count, zero non-finite count, and at most the first two
positive norms. Zero-gradient or zero-step arm evidence must fail.
Also bind an actual schema-2 training length-manifest path/hash and reject a
dry-run zero hash, stale/mutated manifest, wrong retained source, or cross-arm
manifest hash disagreement in the ablation reporter tests.

- [ ] **Step 2: Run tests to verify RED**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_packed_capacity_probe.py \
  tests/step_preference/test_negative_generation.py \
  tests/step_preference/test_packed_capacity_evidence.py \
  tests/step_preference/test_stage2b_trainer.py \
  tests/stage2/test_launcher.py \
  tests/stage2/test_eval_integration.py \
  tests/stage2/test_attention_backend.py \
  tests/step_preference/test_packed_capacity_gate.py \
  tests/step_preference/test_packed_admission_protocol.py \
  tests/step_preference/test_dora_ablation_reporting.py
```

- [ ] **Step 3: Implement the report additions and one validator**

Reset peaks immediately before `Stage2BTrainingStep`, retain the baseline in the
reported peak, synchronize after backward, and compute the ratio from CUDA total
memory. The validator reads once, rejects duplicate JSON keys/non-finite
constants, authenticates all files by SHA-256, and publishes only one summary;
do not add retries or a recovery layer.

Wrap the capacity-only collator with a minimal recorder in
`capacity_evidence.py`; with workers fixed to zero it appends the returned
batch's authenticated `sample_id` on each rank. Add one private exact-sample
selector to the full-admission preflight that runs only
when both capacity evidence and capacity sample ID are present, requires one
matching retained sample, and passes that one-element dataset to the Trainer
without rewriting the shared source or negative artifacts. Extend the
already-installed
`_Stage2DeepSpeedGradientAdapter`: at every true accumulation boundary, after
`collect_deepspeed_pre_step_norm` observes the reduced/unscaled ZeRO-2 gradient
and immediately before `original_step`, append that float to
`trainer._observed_pre_step_gradient_norms`. No post-training parameter-gradient
scan is evidence for this gate. Immediately before `trainer.train()`,
synchronize each CUDA device, record its baseline allocator counters, and reset
peak-memory statistics. After `trainer.train()` has completed step 2,
synchronize again and capture rank-local peak-reserved and total-memory bytes.
Invoke `trainer.save_model(args.save_model_path)` on every rank so
Trainer/DeepSpeed performs its required collective work, barrier, let only
world-process-zero publish the processor files, and barrier again. Only then
publish the collective training-runtime identity, and only then collect
each rank's `(rank,sample_ids,pre_step_gradient_norms,training_loss,
gpu_identity,peak_reserved_bytes,total_memory_bytes,error)` with
`torch.distributed.all_gather_object`. Every rank requires exactly two positive
finite norms, four matching sample observations, one finite loss, an exact
peak-reserved fraction at most 0.85, and no worker error; derive the four health
booleans from those predicates. Have rank zero write the exclusive evidence
file, then barrier.
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
dictionaries. For every final source index, match exactly one distributed
evidence and checkpoint, re-run `inspect_stage2_artifact`, require Stage 2 DoRA,
locate and authenticate exactly `adapter_config.json`,
`adapter_model.safetensors`, and `stage2_peft_metadata.json`, and store their
`(relative_path,size,sha256)` in that source's `distributed_witnesses` entry.
Authenticate every single-device and distributed report against the manifest's
complete final witness set and source-index order.

Implement the fixed one-shot protocol driver described above. It validates the
pilot, Git cleanliness, fresh targets, and live GPU inventory before allocating
the run result/log directories; thereafter it is the sole owner of Task 7
subprocess construction and command provenance. This removes any dependence on
shell-local arrays crossing processes.
Add the minimal negative-generation seed/greedy identity described above; do not
introduce sampling, retry generation, or a second decoding path.

Implement the five-subcommand ablation reporter described above using the same
strict/canonical/exclusive helpers. `freeze-launches` is the only Task 8 writer
allowed before training; `publish-comparison` recomputes metrics through
`EvaluationScoreSummary` and never trusts a supplied rank or accuracy.

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
  tests/step_preference/test_negative_generation.py \
  tests/step_preference/test_packed_capacity_evidence.py \
  tests/step_preference/test_stage2b_trainer.py \
  tests/stage2/test_launcher.py \
  tests/stage2/test_eval_integration.py \
  tests/stage2/test_attention_backend.py \
  tests/step_preference/test_packed_capacity_gate.py \
  tests/step_preference/test_packed_admission_protocol.py \
  tests/step_preference/test_dora_ablation_reporting.py
git add src/step_preference/capacity_evidence.py src/main.py src/stage2/data.py \
  src/stage2/trainer.py \
  src/utils_deepseed.py run_stage2_training.sh eval.py \
  transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py \
  scripts/probe_packed_stage2b_capacity.py \
  scripts/generate_step_negatives.py \
  scripts/run_packed_admission_protocol.py \
  scripts/validate_packed_capacity_gate.py \
  scripts/publish_dora_ablation.py \
  tests/step_preference/test_packed_capacity_probe.py \
  tests/step_preference/test_negative_generation.py \
  tests/step_preference/test_packed_capacity_evidence.py \
  tests/step_preference/test_stage2b_trainer.py \
  tests/stage2/test_launcher.py \
  tests/stage2/test_eval_integration.py \
  tests/stage2/test_attention_backend.py \
  tests/step_preference/test_packed_capacity_gate.py \
  tests/step_preference/test_packed_admission_protocol.py \
  tests/step_preference/test_dora_ablation_reporting.py
git commit -m "test: validate packed dora capacity"
```

---

### Task 7: Materialize and probe the full capacity-admitted cohort

**Files:**

- Write: `results/step-pref-packed-admission-<UTC>/`
- Write: `logs/step-pref-packed-admission-<UTC>/`
- Write: `checkpoints/step-pref-packed-admission-<UTC>-zero2-<source-index>/`

**Interfaces:**

- Consumes: Tasks 1-6, original TRAIN, frozen Stage 2A, and the v13 cap.
- Produces: clean manifest/subset, one shared schema-3 negative artifact,
  admission manifest/retained source, source-recomputed verifier result, all boundary
  capacity probes, one exact two-step accumulation-2 ZeRO-2
  checkpoint/evidence pair per boundary witness, and one pass summary.

- [ ] **Step 1: Run one focused implementation suite and choose a fresh ID**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference tests/stage2
test -z "$(git diff --name-only)"
test -z "$(git diff --cached --name-only)"
git check-ignore -q --no-index results/.codex-ignore-probe
git check-ignore -q --no-index logs/.codex-ignore-probe
git check-ignore -q --no-index checkpoints/.codex-ignore-probe
pilot_summary="${PILOT_SUMMARY_PATH:?absolute Task 1 pilot path required}"
case "$pilot_summary" in /*) ;; *) exit 2 ;; esac
test ! -L "$pilot_summary"
pilot_summary_sha=$(sha256sum "$pilot_summary" | cut -d' ' -f1)
admission_id="step-pref-packed-admission-$(date -u +%Y%m%d-%H%M%S)"
test ! -e "results/$admission_id"
test ! -e "logs/$admission_id"
```

Expected: focused suite passes once, tracked tree is clean, and the candidate
paths are absent. Do not create them here. The protocol driver first requires
the selected premise summary's exact schema/status pass, candidate cap 4608,
frozen TRAIN/checkpoint identities, rejected-v11 chain,
`tracked_worktree_clean=true`, and H100 NVL identity; only then may it allocate
the run result/log directories.

Steps 2-7 below are the fixed logical stages of one driver invocation. The
displayed commands specify the exact subprocess argv/environment contracts and
must not be run a second time by hand. Start the one-shot driver now:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  scripts/run_packed_admission_protocol.py \
  --run-id "$admission_id" \
  --pilot-summary "$pilot_summary" \
  --pilot-summary-sha256 "$pilot_summary_sha"
```

“Reserve” always means require absence; only the named owner creates a path.
The driver creates `results/$admission_id` and `logs/$admission_id` after its
preflight, and owns exact logs `clean-length-manifest.log`, `generate.log`,
`packed-admission-build.log`, `packed-admission-verify.log`,
`witness-<source>-capacity.log`, and `capacity-validator.log` under that log
root, plus the final `capacity-summary-verify.log`. Each two-rank launcher alone creates
`checkpoints/$admission_id-zero2-<source>`,
`results/$admission_id-zero2-<source>`, and
`logs/$admission_id-zero2-<source>/train.log`. The external evidence JSON stays
under the driver's result root as specified below.

- [ ] **Step 2: Build the clean-4608 manifest and exact subset**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  -m stage2.length_manifest \
  --data-path data/TRAIN.jsonl --image-root data \
  --processor-source checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
  --latent-size 8 --supervise-latent-boundaries \
  --max-sequence-length 4608 \
  --assistant-plan-rendering adjacent-text-space-lf-v1 \
  --manifest-path "results/$admission_id/clean-length-manifest.json"
```

Inside the already-running venv driver, read the strict manifest, require `3354`
retained rows and `96` filtered rows, and call
`scripts.select_jsonl_rows.select_jsonl_rows` with its retained indices to
publish `clean-admissible.jsonl`. In the same command, publish canonical
`clean-source-map.json` with the original TRAIN hash, subset hash, exact
retained indices, and canonical prepared sample IDs. Record all three SHA-256
values.

- [ ] **Step 3: Generate the shared negatives exactly once**

```bash
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/python scripts/generate_step_negatives.py \
  --data_path "results/$admission_id/clean-admissible.jsonl" --image_root data \
  --stage2a_checkpoint_path checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
  --output_path "results/$admission_id/negatives.jsonl" \
  --manifest_path "results/$admission_id/negative-manifest.json" \
  --latent_size 8 --min_tokens 2 --max_new_tokens 32 \
  --seed 42 \
  --attention_implementation sdpa \
  --vision_attention_implementation flash_attention_2 --device cuda:0
```

The driver captures stdout/stderr directly to
`logs/$admission_id/generate.log`. Expected: one all-or-nothing schema-3
artifact for 3354 source rows.

- [ ] **Step 4: Build and verify packed admission**

Run `python -m step_preference.packed_admission build` with the paths from Steps
2-3, original TRAIN, image root, checkpoint, cap 4608, and outputs
`packed-retained.jsonl` and `packed-admission.json`. Then hash the manifest and
run `verify` to `packed-admission-verification.json`; hash that result. Require
verifier status pass, nonempty retained rows, `filtered < 34`, and final witness
count at most eight. The `<34` gate is the approved maximum of 33 packed-stage
exclusions, i.e. strictly less than 1% of the 3354 clean-admissible cohort; the
96 clean-prefilter exclusions are reported separately and are not included in
that bound. Tee build/verify logs under `logs/$admission_id/`.
The real witness cardinality is intentionally not assumed in advance. If the
deterministic union contains more than eight, build fails before any capacity
probe and the controller returns to the design-commitment gate; it must not
silently truncate or choose a caller-selected subset.

```bash
clean_manifest_sha=$(sha256sum "results/$admission_id/clean-length-manifest.json" | cut -d' ' -f1)
source_map_sha=$(sha256sum "results/$admission_id/clean-source-map.json" | cut -d' ' -f1)
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
  --max-packed-sequence-length 4608 \
  --retained-output-path "results/$admission_id/packed-retained.jsonl" \
  --manifest-output-path "results/$admission_id/packed-admission.json"
admission_sha=$(sha256sum "results/$admission_id/packed-admission.json" | cut -d' ' -f1)
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  -m step_preference.packed_admission verify \
  --manifest-path "results/$admission_id/packed-admission.json" \
  --manifest-sha256 "$admission_sha" \
  --output-path "results/$admission_id/packed-admission-verification.json"
verification_sha=$(sha256sum "results/$admission_id/packed-admission-verification.json" | cut -d' ' -f1)
```

- [ ] **Step 5: Probe every final boundary witness sequentially**

Immediately before the first probe, require both GPUs idle and have the driver
use one venv process around the nvidia-smi/CUDA inventory pair defined by
`PackedGpuSnapshot` to exclusively publish canonical `gpu-snapshot.json`; record
`gpu_snapshot_sha`. Before every later witness and distributed launch, repeat
both inventories and require the same index/UUID/name/CUDA-total mapping. Set
`CUDA_VISIBLE_DEVICES=0` for each single-device probe and pass `--gpu-snapshot`
plus its SHA-256.

For each manifest witness, run the canonical DoRA capacity probe against the
shared `clean-admissible.jsonl` and full shared negative artifact, passing the
recorded `--sample_id` so `load_production_context` selects exactly that row.
Run it on idle physical GPU 0 and store
`witness-<source-index>-capacity.json` and its log. Require status pass and peak
reserved fraction at most 0.85 before continuing to the next witness.

For each ordered `(source,sample_id)`, the driver executes exactly this argv
(placeholders are substituted as single tokens, and stdout/stderr goes to the
owned witness log):

```text
/usr/bin/env
CUDA_VISIBLE_DEVICES=0
PYTHONPATH=<repo>/transformers/src:<repo>/src
TOKENIZERS_PARALLELISM=false
<repo>/venv/ilvr-venv/bin/python
<repo>/scripts/probe_packed_stage2b_capacity.py
--checkpoint_path
<repo>/checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827
--data_path
<repo>/results/<admission-id>/clean-admissible.jsonl
--image_root
<repo>/data
--negative_path
<repo>/results/<admission-id>/negatives.jsonl
--negative_manifest
<repo>/results/<admission-id>/negative-manifest.json
--output_path
<repo>/results/<admission-id>/witness-<source>-capacity.json
--source_row_index
<source>
--sample_id
<sample-id>
--device
cuda:0
--attention_implementation
sdpa
--vision_attention_implementation
flash_attention_2
--tuning_method
dora
--gradient_checkpointing
--gpu-snapshot
<repo>/results/<admission-id>/gpu-snapshot.json
--gpu-snapshot-sha256
<gpu-snapshot-sha256>
--max_peak_reserved_fraction
0.85
```

The driver reads the exact-field manifest once, materializes the ordered
`(source_index,sample_id)` tuples in memory, rejects control characters in IDs,
and constructs each probe argv directly from those tuples. The same tuple list
is consumed by Step 6; no shell text or log is parsed.

- [ ] **Step 6: Gate every final witness under exact production two-rank ZeRO-2**

Read the manifest's exact ordered `final_probe_source_indices`, resolve each
record's sample ID, and launch one independent capacity run per witness. Reuse
the full authenticated `packed-retained.jsonl`, shared negative artifact,
admission manifest, and verifier result. Set the private capacity sample ID so
preflight selects exactly that already-retained sample in memory; do not create
or regenerate one-row source or negative artifacts.

Each launch uses `STAGE2_NUM_PROCESSES=2`, batch 1, production-identical
accumulation 2, `--max_steps 2 --epochs 1`, combined-positive `(1,1,0.1)`, DoRA
32/64/0.05, checkpointing, SDPA/FA2/Flex, and
`configs/deepspeed_zero2.json`. Before every launch, require both physical GPUs
idle and matching the frozen snapshot. Reserve fresh paths
`checkpoints/$admission_id-zero2-$source`,
`results/$admission_id/zero2-$source-evidence.json`, and
`logs/$admission_id-zero2-$source/train.log` before model execution. The driver
owns `logs/$admission_id/`; each launcher owns its distinct fresh log directory,
matching the launcher's exclusive-directory contract.

```bash
declare -a distributed_evidence_args=()
declare -a distributed_checkpoint_args=()
for source in "${final_probe_source_indices[@]}"; do
  sample_id="${sample_id_by_source[$source]}"
  test ! -e "checkpoints/$admission_id-zero2-$source"
  test ! -e "results/$admission_id/zero2-$source-evidence.json"
  test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)"
  CUDA_VISIBLE_DEVICES=0,1 \
  PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  TOKENIZERS_PARALLELISM=false \
  STAGE1_CHECKPOINT="$PWD/checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827" \
  STAGE2A_CHECKPOINT_PATH="$PWD/checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827" \
  DATA_PATH="$PWD/results/$admission_id/packed-retained.jsonl" \
  IMAGE_ROOT="$PWD/data" \
  OUTPUT_DIR="$PWD/checkpoints/$admission_id-zero2-$source" \
  LOG_FILE="$PWD/logs/$admission_id-zero2-$source/train.log" \
  STEP_PACKED_CAPACITY_EVIDENCE_PATH="$PWD/results/$admission_id/zero2-$source-evidence.json" \
  STEP_PACKED_CAPACITY_SAMPLE_ID="$sample_id" \
  STEP_PACKED_GPU_SNAPSHOT="$PWD/results/$admission_id/gpu-snapshot.json" \
  STEP_PACKED_GPU_SNAPSHOT_SHA256="$gpu_snapshot_sha" \
  STEP_PREFERENCE_ENABLED=true STEP_PREFERENCE_MODE=stage2b \
  STEP_PREFERENCE_PACKING=true STEP_MAX_PACKED_SEQUENCE_LENGTH=4608 \
  STEP_NEGATIVE_PATH="$PWD/results/$admission_id/negatives.jsonl" \
  STEP_NEGATIVE_MANIFEST="$PWD/results/$admission_id/negative-manifest.json" \
  STEP_PACKED_ADMISSION_MANIFEST="$PWD/results/$admission_id/packed-admission.json" \
  STEP_PACKED_ADMISSION_MANIFEST_SHA256="$admission_sha" \
  STEP_PACKED_ADMISSION_VERIFICATION="$PWD/results/$admission_id/packed-admission-verification.json" \
  STEP_PACKED_ADMISSION_VERIFICATION_SHA256="$verification_sha" \
  CE_WEIGHT=1 STEP_DPO_WEIGHT=1 STEP_ENTROPY_WEIGHT=0.1 \
  STEP_DPO_BETA=0.1 STEP_ENTROPY_MARGIN=0.1 \
  TUNING_METHOD=dora LORA_R=32 LORA_ALPHA=64 LORA_DROPOUT=0.05 \
  LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  DEEPSPEED_CONFIG="$PWD/configs/deepspeed_zero2.json" LEARNING_RATE=5e-5 SEED=42 \
  PER_DEVICE_TRAIN_BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=2 GRADIENT_CHECKPOINTING=true \
  ATTENTION_IMPLEMENTATION=sdpa VISION_ATTENTION_IMPLEMENTATION=flash_attention_2 \
  MAX_SEQUENCE_LENGTH=8192 SUPERVISE_LATENT_BOUNDARIES=true \
  DATALOADER_NUM_WORKERS=0 STAGE2_NUM_PROCESSES=2 \
  ./run_stage2_training.sh --max_steps 2 --epochs 1 --save_steps 2
  distributed_evidence_args+=(
    --distributed-evidence
    "results/$admission_id/zero2-$source-evidence.json"
  )
  distributed_checkpoint_args+=(
    --distributed-checkpoint
    "checkpoints/$admission_id-zero2-$source"
  )
done
```

Require every rank to record the same sample on four microbatches, two finite
positive pre-step gradients, two completed optimizer steps, finite losses,
peak-reserved fraction at most 0.85, no skipped update/OOM, positive executed
SDPA/FA2/Flex counters, and a valid final Stage 2 DoRA artifact. A failure stops
the loop and preserves that run; no later witness or arm starts.

- [ ] **Step 7: Publish the capacity decision**

Run `scripts/validate_packed_capacity_gate.py` over the verified admission,
every single-device capacity JSON, and every two-rank evidence/checkpoint pair.
Require overall
pass. Record all commands, hashes, counts, exclusions, GPU identities, and the
capacity-admitted-subset limitation in
`results/$admission_id/capacity-summary.json`.
First exclusively publish the exact `packed-command-log.json` from the collected
argv arrays and compute `commands_sha`.

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  scripts/validate_packed_capacity_gate.py publish \
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
  "${distributed_evidence_args[@]}" \
  "${distributed_checkpoint_args[@]}" \
  --output-path "results/$admission_id/capacity-summary.json"
```

The driver then starts a fresh venv process (not an in-process function call),
captures it in `logs/$admission_id/capacity-summary-verify.log`, and runs:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  scripts/validate_packed_capacity_gate.py verify \
  --summary "results/$admission_id/capacity-summary.json" \
  --summary-sha256 "$(sha256sum "results/$admission_id/capacity-summary.json" | cut -d' ' -f1)" \
  --gpu-snapshot "results/$admission_id/gpu-snapshot.json" \
  --gpu-snapshot-sha256 "$gpu_snapshot_sha"
```

It requires single-device and distributed source-index sets both equal the
manifest's exact final probe set; every rank fraction at most 0.85; every
checkpoint/runtime identity hash reopens; counts
`original=3450,clean_admissible=3354`; and `filtered < 34`. Expected:
`VERIFIED_CAPACITY_V13 status=pass cap=4608`.
After that verifier succeeds, the driver prints exactly
`ADMISSION_RESULT_PATH=<absolute result root> CAPACITY_SUMMARY_SHA256=<digest>`;
the root controller records the path for Task 8. This is the only cross-task
handoff; Task 8 reopens and authenticates every needed hash itself.

---

### Task 8: Train and evaluate CE-only, DPO-only, and entropy-only

**Files:**

- Write: `checkpoints/step-pref-dora-{ce,dpo,entropy}-<UTC>/`
- Write: `results/step-pref-dora-{ce,dpo,entropy}-<UTC>/`
- Write: `logs/step-pref-dora-{ce,dpo,entropy}-<UTC>/`
- Write: `results/step-pref-dora-comparison-<UTC>/{launch-config-ce.json,launch-config-dpo.json,launch-config-entropy.json,comparison.json,comparison.md}`

**Interfaces:**

- Consumes: Task 7 pass summary and its shared retained source, negatives,
  admission manifest, verifier result, and hashes.
- Produces: three complete two-epoch DoRA artifacts, three 400-row evaluations,
  and one comparison ranked by coverage accuracy.

- [ ] **Step 1: Freeze normalized launch records**

Require both GPUs idle, the tracked tree clean, and current commit equal the
Task 7 admission and every witness training-runtime identity before choosing
the ablation stamp. The controller supplies
`ADMISSION_RESULT_PATH` as the absolute, non-symlink result directory printed by
the completed Task 7 driver; Task 8 does not inherit any Task 7 shell variable.
Strictly validate its run-ID grammar/prefix and capacity summary, then recompute
`admission_sha` and `verification_sha` from the authenticated files on disk.
Choose `comparison_id=step-pref-dora-comparison-<UTC>` but
do not create it or any arm directory until all three JSON dry runs validate.

```bash
shared="${ADMISSION_RESULT_PATH:?absolute completed Task 7 result required}"
case "$shared" in /*) ;; *) exit 2 ;; esac
test ! -L "$shared"
admission_id=${shared##*/}
case "$admission_id" in step-pref-packed-admission-*) ;; *) exit 2 ;; esac
capacity_summary="$shared/capacity-summary.json"
capacity_summary_sha=$(sha256sum "$capacity_summary" | cut -d' ' -f1)
admission_sha=$(sha256sum "$shared/packed-admission.json" | cut -d' ' -f1)
verification_sha=$(sha256sum "$shared/packed-admission-verification.json" | cut -d' ' -f1)
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  scripts/validate_packed_capacity_gate.py verify \
  --summary "$capacity_summary" \
  --summary-sha256 "$capacity_summary_sha" \
  --gpu-snapshot "$shared/gpu-snapshot.json" \
  --gpu-snapshot-sha256 "$(sha256sum "$shared/gpu-snapshot.json" | cut -d' ' -f1)"
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  scripts/publish_dora_ablation.py preflight --phase freeze \
  --capacity-summary "$capacity_summary" \
  --capacity-summary-sha256 "$capacity_summary_sha"
```

Use one UTC stamp and, after dry-run validation, create only the fresh
`results/$comparison_id` directory and write canonical
`launch-config-{ce,dpo,entropy}.json` there. Do not precreate an arm's
`results/$run_id`, `logs/$run_id`, or `checkpoints/$run_id`; the launcher owns
those exclusive directories. Normalize run identity paths and coefficient
triplet; assert every other token is identical. Triplets are CE `(1,0,0)`, DPO
`(0,1,0)`, and entropy `(0,0,1)`. Bind the Task 7 capacity summary and all
shared artifact hashes. Compute and retain each exclusive launch config's
SHA-256 before its real launch.

Use this one launcher function for dry-run normalization and all real arms:

```bash
ablation_stamp=$(date -u +%Y%m%d-%H%M%S)
launch_arm() {
  arm=$1; ce=$2; dpo=$3; entropy=$4
  run_id="step-pref-dora-$arm-$ablation_stamp"
  PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
    scripts/publish_dora_ablation.py preflight --phase training --arm "$arm" \
    --capacity-summary "$capacity_summary" \
    --capacity-summary-sha256 "$capacity_summary_sha"
  STAGE1_CHECKPOINT="$PWD/checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827" \
  STAGE2A_CHECKPOINT_PATH="$PWD/checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827" \
  DATA_PATH="$shared/packed-retained.jsonl" IMAGE_ROOT="$PWD/data" \
  OUTPUT_DIR="$PWD/checkpoints/$run_id" LOG_FILE="$PWD/logs/$run_id/train.log" \
  STEP_PREFERENCE_ENABLED=true STEP_PREFERENCE_MODE=stage2b \
  STEP_PREFERENCE_PACKING=true STEP_MAX_PACKED_SEQUENCE_LENGTH=4608 \
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
  SEED=42 \
  PER_DEVICE_TRAIN_BATCH_SIZE=1 \
  GRADIENT_ACCUMULATION_STEPS=2 GRADIENT_CHECKPOINTING=true \
  ATTENTION_IMPLEMENTATION=sdpa VISION_ATTENTION_IMPLEMENTATION=flash_attention_2 \
  MAX_SEQUENCE_LENGTH=8192 SUPERVISE_LATENT_BOUNDARIES=true \
  DATALOADER_NUM_WORKERS=0 STAGE2_NUM_PROCESSES=2 \
  ./run_stage2_training.sh --epochs 2 --warm_up_steps 10 --save_steps 612
}
```

Invoke it with `STAGE2_DRY_RUN=true STAGE2_DRY_RUN_FORMAT=json` for all three
triplets and parse stdout directly as one JSON string array. Replace only the
values following
`--save_model_path`, `--log_file`, `--stage2_length_manifest`, `--ce_weight`,
`--step_dpo_weight`, and `--step_entropy_weight` with respectively
`<CHECKPOINT_DIR>`, `<LOG_FILE>`, `<LENGTH_MANIFEST>`, `<CE_WEIGHT>`,
`<DPO_WEIGHT>`, and `<ENTROPY_WEIGHT>`. Require each option appears exactly once,
each has one following value, and the normalized token lists are equal before
any real launch. The separate `run_id` launch-config field preserves the actual
ID. After replacement, fail if any normalized token still contains any of the
three run IDs or their output paths; no other option or positional token may
vary. The stored `normalized_command` is this JSON token array, never shell
text.

```bash
comparison_id="step-pref-dora-comparison-$ablation_stamp"
ce_run_id="step-pref-dora-ce-$ablation_stamp"
dpo_run_id="step-pref-dora-dpo-$ablation_stamp"
entropy_run_id="step-pref-dora-entropy-$ablation_stamp"
ce_command_json=$(STAGE2_DRY_RUN=true STAGE2_DRY_RUN_FORMAT=json launch_arm ce 1 0 0)
dpo_command_json=$(STAGE2_DRY_RUN=true STAGE2_DRY_RUN_FORMAT=json launch_arm dpo 0 1 0)
entropy_command_json=$(STAGE2_DRY_RUN=true STAGE2_DRY_RUN_FORMAT=json launch_arm entropy 0 0 1)
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  scripts/publish_dora_ablation.py freeze-launches \
  --comparison-dir "$PWD/results/$comparison_id" \
  --capacity-summary "$capacity_summary" \
  --capacity-summary-sha256 "$capacity_summary_sha" \
  --ce-run-id "$ce_run_id" --ce-command-json "$ce_command_json" \
  --dpo-run-id "$dpo_run_id" --dpo-command-json "$dpo_command_json" \
  --entropy-run-id "$entropy_run_id" \
  --entropy-command-json "$entropy_command_json"
```

Expected: only `results/$comparison_id` is created, containing three strict
launch configs; all arm checkpoint/result/log directories remain absent.

- [ ] **Step 2: Launch CE-only and validate its final artifact**

Invoke `run_stage2_training.sh` with the Task 7 packed-retained source and shared
artifacts, `TUNING_METHOD=dora`, rank/alpha/dropout 32/64/0.05,
`GRADIENT_ACCUMULATION_STEPS=2`, `MAX_SEQUENCE_LENGTH=8192`, packed cap 4608,
checkpointing true, SDPA/FA2, ZeRO-2, seed 42, two epochs, and CE coefficients
`1,0,0`. Require exit zero, finite metrics, no skipped update/OOM, and
`inspect_stage2_artifact` classification Stage 2 DoRA. Authenticate the final
training-runtime sidecar and require actual SDPA/FA2 classes plus a positive
FlexAttention count for every language layer, positive/equal global and
optimizer steps, the matching finite active-objective metric, at least one
finite positive pre-step gradient, and zero non-finite gradient observations.
Immediately before each arm,
require both GPUs have no compute applications and still match the Task 7 GPU
snapshot; do not signal or reuse any foreign process.

```bash
launch_arm ce 1 0 0
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  scripts/publish_dora_ablation.py validate-arm --phase training \
  --comparison-dir "$PWD/results/$comparison_id" --arm ce \
  --capacity-summary "$capacity_summary" \
  --capacity-summary-sha256 "$capacity_summary_sha"
```

- [ ] **Step 3: Launch DPO-only, then entropy-only, with the same validation**

Run sequentially, changing only the run identity and coefficient triplet to
`0,1,0` and then `0,0,1`. A failed arm stops later arms. Do not resume a partial
run; diagnose, add one causal test/fix, and use a fresh run ID.

The required causal regression/fix changes the tracked Git identity, so any arm
or post-model-construction evaluation failure invalidates the Task 7 capacity
summary and every arm. Commit the causal
test/fix, rerun Task 7 under a fresh admission ID and new ablation stamp, then
rerun CE/DPO/entropy from the beginning; never mix an earlier successful arm
with post-fix arms. Any proposed change to cap, population, objective, or
comparison semantics is outside commitment v13 and must return to the
design-commitment approval gate rather than being silently treated as a retry.

```bash
launch_arm dpo 0 1 0
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  scripts/publish_dora_ablation.py validate-arm --phase training \
  --comparison-dir "$PWD/results/$comparison_id" --arm dpo \
  --capacity-summary "$capacity_summary" \
  --capacity-summary-sha256 "$capacity_summary_sha"
launch_arm entropy 0 0 1
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  scripts/publish_dora_ablation.py validate-arm --phase training \
  --comparison-dir "$PWD/results/$comparison_id" --arm entropy \
  --capacity-summary "$capacity_summary" \
  --capacity-summary-sha256 "$capacity_summary_sha"
```

- [ ] **Step 4: Evaluate each final adapter on all 400 TEST rows**

Run `eval.py` with the frozen Stage 2A base and each DoRA adapter, TEST hash
`6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618`,
task `comt`, language SDPA, vision FlashAttention2,
`max_sequence_length=32768`, `max_new_tokens=4096`, repetition penalty 1.05,
temperature 0, and top-p 1. Require exactly 400 sample records plus one summary
with finite coverage accuracy, valid-gold accuracy, invalid-gold count, and
inference time, and require zero inference-error rows. A caught per-row exception
preserves diagnostic output but fails that evaluation and triggers the runtime-
failure rule; it is never ranked as merely an incorrect prediction. Immediately
before each sequential evaluation, require physical
GPU 0 idle and matching the snapshot; validate its output and runtime identity
before starting the next arm.
`eval.py` intentionally has no backend CLI: Task 6 tests and preserves its
hard-coded `load_qwen2_5_vl_with_attention(..., attention_implementation="sdpa",
vision_attention_implementation="flash_attention_2")`, and the exclusive
runtime-identity sidecar observes the loaded concrete classes before inference.

```bash
evaluate_arm() {
  arm=$1
  run_id="step-pref-dora-$arm-$ablation_stamp"
  PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
    scripts/publish_dora_ablation.py preflight --phase evaluation --arm "$arm" \
    --capacity-summary "$capacity_summary" \
    --capacity-summary-sha256 "$capacity_summary_sha"
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
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  scripts/publish_dora_ablation.py validate-arm --phase evaluation \
  --comparison-dir "$PWD/results/$comparison_id" --arm ce \
  --capacity-summary "$capacity_summary" \
  --capacity-summary-sha256 "$capacity_summary_sha" \
  --test-data "$PWD/data/TEST.jsonl" \
  --test-data-sha256 6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618
evaluate_arm dpo
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  scripts/publish_dora_ablation.py validate-arm --phase evaluation \
  --comparison-dir "$PWD/results/$comparison_id" --arm dpo \
  --capacity-summary "$capacity_summary" \
  --capacity-summary-sha256 "$capacity_summary_sha" \
  --test-data "$PWD/data/TEST.jsonl" \
  --test-data-sha256 6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618
evaluate_arm entropy
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  scripts/publish_dora_ablation.py validate-arm --phase evaluation \
  --comparison-dir "$PWD/results/$comparison_id" --arm entropy \
  --capacity-summary "$capacity_summary" \
  --capacity-summary-sha256 "$capacity_summary_sha" \
  --test-data "$PWD/data/TEST.jsonl" \
  --test-data-sha256 6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618
```

- [ ] **Step 5: Publish the comparison and completion evidence**

Inspect all three adapters; validate three evaluation cardinalities and hashes;
authenticate each runtime-identity sidecar and its exact SDPA/FlashAttention2
class names/counts; hash each training-runtime sidecar, predictions JSONL, and
evaluation-runtime sidecar into that run's exact comparison fields; recheck
each comparison-directory launch-config hash, run ID, triplet, normalized
command, and shared artifact hashes. Authenticate launch config and runtime
identity separately, then cross-check their overlapping run ID, coefficient,
admission/verifier hashes, and runtime-setting fields; check Git against the
admission identity separately. This is a consistency check, not
a claimed direct hash chain. Require every Git commit equals the admission
identity and tracked-clean is true;
recheck normalized config equality; rank by `correct/400`, preserving exact ties
and using valid-gold accuracy only as secondary reporting. Write canonical
`results/$comparison_id/comparison.json` and
`results/$comparison_id/comparison.md`. State prominently that the
training comparison applies only to the capacity-admitted subset while TEST
evaluation covers all 400 rows.

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  scripts/publish_dora_ablation.py publish-comparison \
  --comparison-dir "$PWD/results/$comparison_id" \
  --capacity-summary "$capacity_summary" \
  --capacity-summary-sha256 "$capacity_summary_sha" \
  --test-data "$PWD/data/TEST.jsonl" \
  --test-data-sha256 6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618
```

- [ ] **Step 6: Run final verification once**

Run the full project suite once:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q
```

Then run only distinct artifact/evaluation/config validators needed for the
completion claims. The per-arm non-writing commands in Steps 2-4 already ran
each checkpoint/evaluation validator once, so do not repeat them. Run the fresh
aggregate verifier once; it internally calls `inspect_stage2_artifact` for all
three checkpoints, validates each exact 401-line evaluation and both runtime
identities, rechecks normalized/shared hashes, reconstructs ranks/Markdown, and
strictly compares both published outputs:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  scripts/publish_dora_ablation.py verify-comparison \
  --comparison-dir "$PWD/results/$comparison_id" \
  --capacity-summary "$capacity_summary" \
  --capacity-summary-sha256 "$capacity_summary_sha" \
  --test-data "$PWD/data/TEST.jsonl" \
  --test-data-sha256 6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618
```

Expected: one line `VERIFIED_DORA_ABLATION arms=3 test_rows=400` and exit zero.
Record this and the full-suite output before claiming completion.
