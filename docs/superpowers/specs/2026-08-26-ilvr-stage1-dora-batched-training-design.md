# ILVR Stage 1 Batched Full/DoRA Training Design

**Date:** 2026-08-26
**Status:** Proposed for user review
**Repository:** `XD111ds/ILVR`
**Depends on:**

- `2026-08-24-ilvr-stage2-reimplementation-design.md`
- `2026-08-25-ilvr-stage2-peft-design.md`

## 1. Purpose

This design hardens ILVR Stage 1 for two production experiment paths:

```text
Qwen2.5-VL-3B Stage 1 full tuning
Qwen2.5-VL-3B Stage 1 DoRA tuning
```

Both paths must support true rank-local batches greater than one, normal
checkpoint resume, two-rank DeepSpeed ZeRO-3 execution, and a reproducible
three-epoch CoMT timing run. The Stage 1 DoRA artifact must also be a valid
starting point for Stage 2 DoRA without adapter merging or stacking.

The implementation must preserve the paper's existing Stage 1 semantics:

- a frozen visual encoder;
- an EMA teacher;
- helper-image-derived latent targets;
- textual CE plus the existing alignment objective;
- Stage 1 helper images visible to the teacher but not treated as ordinary
  student visual inputs.

This design supersedes the earlier PEFT design only where that design said
Stage 1 behavior would remain unchanged or that Stage 1 PEFT-only artifacts
could not initialize Stage 2. Existing Stage 2 recurrence semantics remain
binding and unchanged.

## 2. Binding Experiment Decisions

The first timing and readiness experiment uses:

```text
model                         Qwen/Qwen2.5-VL-3B-Instruct
dataset                       complete local CoMT TRAIN.jsonl (3,450 samples)
epochs                        3
GPUs                          both local NVIDIA B200 GPUs
world size                    2
methods                       full, dora
learning rate                 1e-5 for both methods
gradient accumulation         2
initial per-GPU micro-batch   8
micro-batch search            8, 16, 32, ... until the first capacity failure
seed                          42
attention                     FlashAttention 2
backend                       the same DeepSpeed ZeRO-3 configuration
intermediate checkpoints      disabled for the timed run
final checkpoint              saved once after training
```

If batch 8 fails, capacity probing descends through 4, 2, and 1. Full and DoRA
are probed independently because their trainable state and optimizer memory
differ. Their final effective global batches may therefore differ:

```text
effective_global_batch = 2 GPUs * per_GPU_micro_batch * 2 accumulation steps
```

The result is a maximum-throughput comparison, not a same-batch controlled
optimization comparison. Reports must state the effective batch and optimizer
step count for each method. A later common-batch control is optional and is not
part of this first run.

DoRA uses:

```text
r                32
alpha            64
dropout          0.05
target suffixes  q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

LoRA is not an experiment variant in this work. The legacy Stage 1
`--use_lora` interface remains compatible, but it is not expanded, benchmarked,
or used for Stage 2 handoff.

## 3. Current Repository Findings

The current Stage 1 path is not safe for the requested experiment:

1. `collate_fn_stage1` flattens multimodal processor outputs without recording
   per-sample helper-image ownership.
2. `CustomTrainerStage1` discovers latent segments only from `input_ids[0]`,
   creates masks for batch row zero, and builds teacher targets as if every
   batch contained one example.
3. The legacy Qwen latent-visual branch explicitly rejects `B != 1`.
4. Stage 1 passes a legacy `LoraConfig` into `SFTTrainer`, but there is no real
   Stage 1 adapter save/resume, ZeRO-3, or EMA-wrapper test.
5. `run_training.sh` does not forward `--use_lora` or PEFT settings.
6. Stage 1's CLI exposes `--learning_rate`, but the trainer configuration
   hard-codes `1e-5` instead of consuming the argument.
7. Stage 1 adds three latent tokens to the base tokenizer. Decoder-only PEFT
   freezes their newly resized input and output rows, leaving random latent
   boundary representations. The legacy LoRA path does not address this.
8. Stage 2 currently rejects a Stage 1 PEFT-only starting artifact.

These are correctness and artifact-lifecycle gaps, not only launcher gaps.

## 4. Supported Training and Transition Matrix

The authoritative initial experiment matrix is:

```text
Stage 1 full  -> Stage 2 full
Stage 1 DoRA  -> Stage 2 DoRA
```

The existing Stage 1 full to Stage 2 DoRA initialization remains a supported
compatibility path, but it is not one of the two pipelines in the first timing
and performance experiment. This work must not regress that path.

The DoRA path continues the same learned adapter state:

```text
base Qwen weights
  + Stage 1 DoRA projection deltas
  + Stage 1 latent-token row deltas
        -> configure Stage 2 semantics
        -> keep the same deltas trainable
        -> create a new Stage 2 optimizer and scheduler
```

Unsupported transitions fail clearly:

- Stage 1 DoRA to Stage 2 full;
- Stage 1 DoRA to a new or stacked Stage 2 adapter;
- Stage 1 DoRA to Stage 2 LoRA;
- changed rank, alpha, dropout, target set, token IDs, or architecture;
- implicit merge into the base model;
- resuming Stage 1 optimizer state as if it were Stage 2 optimizer state.

Stage 1 DoRA resume and Stage 1-to-Stage 2 transition are distinct:

```text
Stage 1 DoRA checkpoint -> resumed Stage 1 DoRA
  restore adapter, token rows, optimizer, scheduler, RNG, and global step

Stage 1 DoRA checkpoint -> fresh Stage 2 DoRA
  restore adapter and token rows only
  reset optimizer, scheduler, RNG policy, and Stage 2 global step
```

## 5. Stage 1 Component Architecture

Stage 1-specific responsibilities move behind explicit components under a new
`src/stage1/` package. `CustomTrainerStage1` remains the Trainer entry point but
delegates target construction and artifact handling.

### 5.1 `Stage1BatchMetadata`

The collator emits ordinary model tensors plus one trainer-owned metadata
object containing, for each sample:

- original image count and corresponding grid range;
- helper image count and corresponding grid range;
- ordered latent-block spans after template replacement;
- ordered helper-image-to-latent-block mapping;
- unpadded sequence length;
- a stable sample identifier for diagnostics.

Dataset preparation attaches the zero-based source JSONL row index as the
stable identifier. If a raw sample already has an identifier, that value is
recorded as supplementary provenance but never replaces the unique row index.

The metadata is removed before calling the student model. It is never accepted
by generic Qwen forward or serialized into model inputs.

The collator validates:

```text
number of latent blocks == number of helper images
every block has exactly latent_size pad slots
blocks are ordered, nonnested, and complete
flattened image grids exactly partition by sample
input_ids, attention_mask, labels, and masks share B and T
```

Malformed samples fail with the sample identifier and the violated count or
span. Cross-sample inference or silent redistribution is forbidden.

### 5.2 `Stage1TeacherTargetBuilder`

The target builder owns EMA-teacher observation and returns:

```python
Stage1TeacherTargets(
    latent_values,
    latent_mask,
    selected_counts,
    diagnostics,
)
```

`latent_mask` has shape `[B, T]`. `latent_values` are ordered exactly like the
true entries of `latent_mask` in row-major order so the student can inject them
without assuming equal latent counts between samples.

For one batch the builder performs:

1. one batched EMA-teacher language forward over original images and text;
2. one batched EMA-teacher visual encoding over all helper images;
3. exact slicing of teacher outputs using `Stage1BatchMetadata`;
4. sample-local attention pooling and helper-feature selection;
5. sample-local sequential block conditioning, including the previous selected
   helper state where required by the current Stage 1 algorithm;
6. row-major packing of selected latent targets and masks.

Sample `b+1` must never observe sample `b`'s text, helper features, selected
latent state, or block-local counters. Multiple blocks in one sample remain
ordered and contextual.

The initial implementation favors semantic parity over a fused optimization.
It may use vectorized batched tensor operations with sample-local masks, but it
must not simulate `B>1` by running a complete student forward once per sample.

### 5.3 Student input boundary

Helper-image tensors are always consumed and removed by the target builder
before the student forward. The official Stage 1 trainer never forwards
`pixel_values_latent` or `image_grid_thw_latent` to the student model.

The student receives:

```text
original image tensors
input_ids / attention_mask / labels
latent_hidden_states
image_out_mask
```

The existing `latent_hidden_states` masked-scatter path is generalized and
validated for row-major batched values. A missing teacher target for a valid
latent block is an error; it does not fall back to direct helper-image encoding
inside the student.

### 5.4 Loss semantics

The mathematical objective remains:

```text
loss = ce_weight * text_CE + sim_weight * latent_alignment
```

Losses are computed across every valid sample and latent position, not only row
zero. Padding and unsupervised latent slots are masked. `B=1` must reproduce the
existing loss and target-selection behavior within numerical tolerance.

No DoRA-specific auxiliary objective is added.

## 6. EMA Teacher and Trainable Boundaries

The EMA teacher is constructed after the final full/DoRA student topology is
known. It is always eval-only and has no gradients.

Parameter correspondence is name- and shape-based rather than positional
`zip`. Initialization and every update fail if teacher/student topology differs.

For full tuning, EMA updates all trainable language parameters. For DoRA, EMA
updates only:

- DoRA direction and magnitude parameters;
- the three trainable latent input-token rows;
- the three trainable latent output-token rows.

Frozen base and visual parameters remain identical and are not repeatedly
updated. Under ZeRO-3, only the required sharded source parameters are gathered
for the update. The implementation must not materialize the complete student
model on every rank merely to update the EMA teacher.

The visual module is frozen for both full and DoRA. Teacher visual computation
runs under `torch.no_grad()`. Student language and DoRA computation remain
differentiable.

## 7. DoRA and Latent Token Rows

### 7.1 Decoder DoRA

DoRA is attached only to full decoder module paths ending in:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Target resolution must prove decoder ownership, require every requested suffix
to match, and reject visual, embedding, LM-head, or unrelated modules.

### 7.2 Sparse latent-token input rows

The three newly added token IDs are recorded before PEFT construction. PEFT's
selective trainable-token mechanism is used for only these input embedding
rows. All pre-existing vocabulary rows remain frozen.

### 7.3 Sparse latent-token output rows

Qwen2.5-VL does not tie its input embedding and LM head. A repository-owned
output-row adapter therefore adds trainable deltas for exactly the three latent
token rows:

```text
delta shape = [3, hidden_size]
```

For normal logits the wrapper computes the frozen base LM head and adds delta
logits only at the three configured token IDs. It does not copy or train the
complete vocabulary projection.

The row adapter must support:

- bf16 training with fp32-safe serialization metadata;
- DeepSpeed ZeRO-3 partitioning and gathering;
- save/load without merging;
- EMA teacher updates;
- Stage 1 resume;
- Stage 2 continuation and inference;
- exact token-ID mismatch validation.

This sparse adapter is the only exception to Stage 2's existing frozen
embedding/LM-head boundary. All ordinary vocabulary rows remain frozen. The
three learned row deltas stay trainable when continuing Stage 1 DoRA into Stage
2 DoRA because Stage 2 must learn when to emit latent boundaries.

## 8. Checkpoint and Metadata Contracts

### 8.1 Stage 1 DoRA artifact

A complete Stage 1 DoRA artifact contains:

```text
adapter_config.json
adapter_model.safetensors
ilvr_token_rows.safetensors
stage1_peft_metadata.json
tokenizer / processor files
Trainer state files when the directory is a numbered checkpoint
```

It contains no merged or copied full base-model weight file.

The versioned Stage 1 sidecar records:

- schema version and `stage = "stage1"`;
- base-model reference;
- the strict Qwen2.5-VL text/vision architecture snapshot;
- base and resized vocabulary sizes;
- all three latent token IDs and token strings;
- `latent_size` and Stage 1 loss/selection settings that affect semantics;
- DoRA rank, alpha, dropout, requested suffixes, and resolved full names;
- sparse input/output row-adapter format and tensor shapes;
- PEFT and repository format versions.

Artifact classification rejects incomplete, mixed full-plus-PEFT, legacy
binary-adapter, missing token-row, and unknown-schema layouts before loading
large weights.

### 8.2 Stage 2 continuation artifact

Stage 2's sidecar schema is extended to state whether latent token rows live in
the selected full Stage 1 base or in an auxiliary sparse adapter:

```text
token_row_source = "base" | "sparse_adapter"
stage1_adapter_reference = string or null, always present
```

For `sparse_adapter`, `ilvr_token_rows.safetensors` is mandatory in the Stage 2
artifact. The final Stage 2 adapter contains the complete current DoRA and token
row deltas and can load with only the original compatible Qwen base. It does
not depend on the continued presence of the Stage 1 artifact path.

Relocation validates structure and token identity without hashing the full base
weight tree. The Stage 2 sidecar advances from the current unreleased schema v2
to schema v3. A strict v2 reader migration is supported only for artifacts that
match the complete v2 field set; it supplies
`token_row_source = "base"` and `stage1_adapter_reference = null`. V1 remains
rejected because it lacks the architecture provenance added in v2. Newly saved
artifacts always use v3, and unknown or partially modified schemas fail closed.

### 8.3 CLI source separation

Fresh Stage 2 DoRA continuation uses explicit sources:

```bash
--load_model_path /path/to/original-qwen-base
--stage1_adapter_path /path/to/stage1-dora-artifact
--tuning_method dora
```

`--resume_from_checkpoint_path` remains reserved for resuming an already
running Stage 2 Trainer checkpoint. Passing both a Stage 1 adapter initializer
and a Stage 2 resume checkpoint is invalid.

## 9. Training Configuration and Launcher

Stage 1 receives a repository-owned launcher rather than requiring users to
edit hard-coded shell variables. It exposes and validates:

- model source;
- tuning method (`full` or `dora` for the official experiment);
- data and image roots;
- output and log paths;
- epochs, learning rate, micro-batch, accumulation, and seed;
- DoRA settings;
- attention implementation;
- DeepSpeed config;
- intermediate/final save policy;
- explicit resume source.

The Python trainer consumes `args.learning_rate`; no second hard-coded learning
rate is allowed.

The legacy `run_training.sh` remains usable, but documentation directs new
full/DoRA runs to the validated launcher. Stage 1 `--use_lora` cannot be
combined with `--tuning_method` and is never silently interpreted as DoRA.

## 10. Capacity Probe and Measurement

A separate benchmark controller launches fresh processes for each candidate
micro-batch. It starts at 8 and probes powers of two. It never retries inside a
dirty CUDA process.

Each candidate executes enough real CoMT work to cover:

- preprocessing and multimodal collation;
- EMA teacher original-image forward;
- helper visual encoding;
- target selection;
- student forward/backward;
- optimizer and scheduler step;
- at least one post-step iteration.

The probe uses high-cost real samples selected by a one-time lightweight
length/image-footprint audit. It records the selection and does not hash or
reread large immutable image trees merely to create evidence.

The first OOM triggers systematic debugging to distinguish expected capacity
exhaustion from shape growth, cross-sample leakage, stale processes, or a
memory leak. After root-cause classification, the previous successful batch is
validated in a fresh process with a longer smoke. No speculative model changes
or silent batch reduction occur inside production training.

The timed three-epoch process records:

- model and processor load time;
- Trainer-only time;
- total wall-clock time;
- final save time;
- samples/second and steps/second;
- rank-local peak allocated and reserved CUDA memory;
- trainable and total parameters;
- effective batch and optimizer step count;
- finite loss and representative gradient norms;
- checkpoint artifact sizes;
- git commit, dependency versions, GPU model/count, and effective config.

Intermediate checkpoint saving is disabled during the timed run. Final save
cost is recorded separately so full-model serialization does not contaminate
Trainer throughput.

## 11. Test-Driven Acceptance Criteria

Every behavior change follows TDD. At minimum, tests cover:

### Batch correctness

- `B=1` target, mask, CE, and alignment parity with the existing path;
- batched target output equals row-major composition of independent samples;
- no cross-sample helper feature, text, mask, or previous-block state leakage;
- variable sequence lengths, original image counts, helper counts, and block
  counts;
- multiple latent blocks per sample;
- one helper visual-encoder call per Stage 1 batch;
- malformed ownership/count/span failures;
- true batched student forward rather than a per-sample forward loop.

### Gradients and EMA

- full language parameters receive finite expected gradients;
- DoRA direction and magnitude parameters receive finite nonzero gradients;
- exactly three input and three output token rows receive gradients;
- ordinary embedding/LM-head rows, base decoder, and visual parameters remain
  frozen in DoRA;
- EMA name/shape matching fails on topology mismatch;
- EMA updates full parameters in full mode and adapter/token rows in DoRA;
- ZeRO-partitioned parameter updates do not use zero-sized tensors as evidence.

### Checkpoint lifecycle

- Stage 1 full final and numbered checkpoint save/resume;
- Stage 1 DoRA adapter-only final and numbered checkpoint save/resume;
- processor/token IDs and architecture metadata round-trip;
- incomplete/mixed/mismatched artifacts fail before expensive loading;
- Stage 1 DoRA to Stage 2 DoRA preserves initial adapter and token-row logits;
- Stage 2 transition resets optimizer/scheduler/global step;
- Stage 2 resume restores them normally;
- final Stage 2 artifact no longer requires the Stage 1 adapter directory;
- direct and DeepSpeed-wrapped inference resolve all active adapters.

### Launcher and experiment evidence

- exact two-GPU commands for full and DoRA;
- micro-batch starts at 8 and follows the approved search order;
- accumulation is exactly 2;
- both learning rates are exactly `1e-5`;
- three epochs and no intermediate timed checkpoints;
- output directories cannot collide between variants or probe attempts;
- one two-rank ZeRO-3 `B>1` save/resume witness for full and DoRA;
- one final three-epoch run per method only after smoke evidence exists.

### Regressions

- the full feasible Stage 1 test suite;
- the existing Stage 2 suite;
- Stage 2 train/inference latent parity;
- generic generation regression;
- Stage 1 legacy `B=1` and `--use_lora` CLI compatibility.

## 12. Error Handling

Fail before expensive training for:

- unsupported or contradictory tuning flags;
- missing or duplicated latent special tokens;
- trainable visual or ordinary vocabulary parameters in DoRA;
- invalid decoder target topology;
- missing sparse token-row state;
- helper-image/block ownership mismatch;
- unsupported artifact combination or schema;
- Stage 1 adapter plus Stage 2 resume supplied together;
- Stage 1/Stage 2 DoRA hyperparameter mismatch;
- rank-local batch below one or malformed distributed rank metadata;
- nonfinite learning rate, dropout, loss, gradient, or timing value.

Broad exception fallbacks are prohibited. FlashAttention fallback behavior
remains the existing traceback-provenance classifier. Unexpected numerical,
OOM, DeepSpeed, wrapper, or integration failures invoke systematic debugging
before fixes.

## 13. Expected File Boundaries

Expected additions:

```text
src/stage1/__init__.py
src/stage1/data.py
src/stage1/targets.py
src/stage1/peft.py
src/stage1/checkpointing.py
src/stage1/benchmark.py
run_stage1_training.sh
run_stage1_batch_probe.sh
tests/stage1/...
```

Expected modifications:

```text
src/main.py
src/trainer.py
src/utils_deepseed.py
src/stage2/peft.py
src/stage2/checkpointing.py
src/stage2/trainer.py
eval.py / src/evaluate_deepseed.py where token-row loading is required
README.md
focused Stage 2 regression tests
```

Generic `GenerationMixin` and the authoritative Stage 2 latent recurrence must
not acquire Stage 1 PEFT or batch logic.

Shared decoder-target, architecture-snapshot, and artifact-classification code
may move to a stage-neutral module only when both stages consume the same
mathematical contract and regression coverage protects existing Stage 2
artifacts.

## 14. Implementation Milestones

### M0 — Contracts and fixtures

Define Stage 1 tuning, metadata, sample ownership, and sparse token-row
interfaces with failing tests.

### M1 — Batched Stage 1 semantics

Implement batch-safe collator metadata, teacher targets, loss, and EMA behavior.
Prove `B=1` parity and multi-sample isolation on tiny deterministic models.

### M2 — Full/DoRA lifecycle

Implement explicit full/DoRA composition, token rows, adapter-only saving,
resume, artifact validation, and parameter-boundary diagnostics.

### M3 — Stage 2 DoRA continuation

Load and continue Stage 1 projection/token-row adapters with fresh Stage 2
training state. Preserve existing full-to-full and full-to-DoRA behavior.

### M4 — Distributed and launcher readiness

Validate two-rank ZeRO-3 `B>1` full and DoRA save/resume, then validate batch
probe behavior starting at 8.

### M5 — Three-epoch experiment

Run the approved full and DoRA configurations exactly once each at their
validated maximum micro-batches. Record timing, memory, throughput, loss, and
artifacts. Do not claim paper-level reproduction.

## 15. Explicit Non-Goals

- Stage 1 LoRA experiments;
- QLoRA or quantized training;
- Stage 2 arbitrary `B>1` scheduling;
- adapter merge, stacking, fusion, or export;
- automatic rank, alpha, dropout, or learning-rate sweeps;
- datasets other than CoMT;
- models other than Qwen2.5-VL-3B for the first experiment;
- changing Stage 2 continuous-latent mathematics;
- paper-level benchmark reproduction from a three-epoch timing run.

## 16. Completion Reporting

The final implementation and experiment report must distinguish:

```text
implementation verified
two-rank B>1 smoke verified
capacity probe verified
three-epoch full run completed
three-epoch DoRA run completed
Stage 2 handoff verified
paper-level reproduction not performed
```

It must list exact commands, commits, effective batches, optimizer steps,
timings, memory, throughput, save sizes, tests, warnings, skipped work, and any
hardware or data limitation. Expensive successful evidence is reused rather
than repeated solely for reporting.
