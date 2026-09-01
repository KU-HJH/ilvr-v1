# Stage 1 True-Batch Equivalence Design

**Date:** 2026-09-01  
**Status:** Design proposal v9, frozen for critique
**Repository:** `XD111ds/ILVR`  
**Supersedes:** only the batch-equivalence portions of
`2026-08-26-ilvr-stage1-dora-batched-training-design.md`

## 1. Goal

Make Stage 1 `B=2, gradient_accumulation=2` training a controlled equivalent
of `B=1, gradient_accumulation=4` at world size four and effective global batch
16 while retaining a true two-row student forward. Every controllable input to
an optimizer update must agree between the two paths:

- the canonical ordered global sample schedule;
- per-sample CE and alignment objectives;
- per-sample DoRA dropout and alignment-noise draws;
- optimizer-step count and boundaries;
- learning-rate scheduler state;
- EMA update count and boundaries.

"Agree" means that these definitions, exogenous inputs, weights, and update
rules agree. It does not require CE, alignment, or teacher values to remain
numerically equal after the two model/EMA states have diverged from the common
first update.

The trainable language decoder remains a true B=2 operation. Frozen visual
features and EMA-teacher targets are computed sample by sample before that
decoder call because an actual 3B probe disproved their batch invariance. The
only permitted differences are GPU floating-point effects from decoder
forward, backward, and reduction execution, including the separately measured
same-shape repeatability floor. Bitwise decoder equality is not a goal because
it would require serial B=1 decoder forwards and remove the requested batching
benefit.

## 2. Fixed Experiment Contract

The controlled comparison is:

```text
model                         Qwen2.5-VL-3B-Instruct
model/processor path          checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct
method                        DoRA
DoRA rank / alpha             32 / 64
DoRA target modules           q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
DeepSpeed                     ZeRO-1
world size                    4
seed                          42
epochs                        3
dataset                       data/TRAIN.jsonl, 3,450 source rows
dataset SHA-256               5bc1313a594f5713975b67388c745c6632275c37fd260055b27230c9cfedbed8
image root                    data/images_comt
canonical epoch slots         3,456
effective global batch        16
reference                     B=1, GA=4
candidate                     B=2, GA=2
optimizer steps per epoch     216
total optimizer steps         648
EMA tau                       0.999
learning rate                 1e-5
optimizer                     adamw_torch_fused
Adam betas / epsilon          0.9, 0.999 / 1e-8
weight decay / max grad norm  0.01 / 1.0
LR scheduler / warmup         linear / 10 optimizer steps
CE weight / alignment weight  1.0 / 1.0
coverage p / image pool k     0.9 / 8
helper group L / latent size  256 / 8
DoRA dropout                  0.05
alignment target noise        0.01
maximum sequence length       32,768
attention                     FlashAttention 2
precision                     BF16 training, FP32 loss reductions
gradient checkpointing        enabled, non-reentrant for DoRA
logging / checkpoints         every 20 steps / epoch boundaries 216,432,648
DeepSpeed config SHA-256      c0922c8640778ae928c5bcacb070f2efaf34b7122431d9b4c2989e25debd
evaluation dataset            data/TEST.jsonl
evaluation dataset SHA-256    6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618
evaluation generation         4 shards, max_new_tokens=4096, temperature=0, top_p=1
```

The comparison does not use an earlier B=2 checkpoint as a B=1 control. Both
fresh paired runs start from the same immutable initial model and tuning state.

Before either witness or full run starts, the launcher materializes a resolved
experiment manifest. The B=1 and B=2 manifests must be identical except for
this explicit whitelist:

```text
per-device microbatch size             1 versus 2
gradient accumulation steps            4 versus 2
derived rank-local microbatch grouping
run ID and checkpoints/results/logs destinations
```

Model and processor revisions, code revision, trainable-parameter names and
initial values, dataset bytes and ordering, optimizer and scheduler settings,
DeepSpeed ZeRO-1 settings other than the derived microbatch fields, EMA
settings, precision, kernels, seeds, loss weights, and evaluation settings are
not whitelisted. Any such manifest difference is a launch error.

### 2.1 Resolved-manifest schema

The implementation writes schema-versioned JSON using UTF-8 canonical JSON
(`sort_keys=True`, separators `,` and `:`, no NaN/Infinity) and publishes its
SHA-256. Semantic configuration and immutable identities live in the manifest;
timestamps, PIDs, hostnames, GPU IDs, and observed timing/memory live only in
runtime evidence. Model/processor files, image files actually referenced by
the dataset, source JSONL, test JSONL, DeepSpeed JSON, and the clean code commit
are represented by SHA-256 digests. Trainable initialization is represented by
the ordered fully-qualified parameter name, dtype, shape, and tensor SHA-256.

The only B=1/B=2 manifest JSON pointers allowed to differ are:

```text
/batch/per_device_train_batch_size
/batch/gradient_accumulation_steps
/batch/rank_local_microbatch_grouping
/artifacts/run_id
/artifacts/checkpoints
/artifacts/results
/artifacts/logs
```

The B=1 A/A replay additionally permits
`/diagnostics/bounded_optimizer_steps=1`. Generated DeepSpeed `auto` values are
resolved before comparison and may differ only when they are the direct values
of the three approved `/batch/` fields. Paths are stored as repository-relative
logical identities where possible; host-specific absolute mirrors are runtime
evidence and cannot mask a semantic digest difference.

## 3. Considered Approaches

### 3.1 Selected: true B=2 decoder with sample-stable semantics

Keep the trainable student decoder batched and make frozen visual inputs,
teacher targets, stochastic draws, objectives, and the global sample schedule
independent of microbatch grouping. This retains decoder batching while making
the remaining numerical difference observable.

### 3.2 Rejected: disable dropout and alignment noise

This is simpler but changes the approved DoRA and ILVR objectives. It would
prove equivalence for a different experiment rather than the requested one.

### 3.3 Rejected: serialize each B=2 row as a B=1 forward

This provides strict gradient equality but is not true batching and forfeits
the memory and throughput behavior under investigation.

### 3.4 Rejected: separate paired short jobs before the main runs

At the first step both paths use the same initial EMA teacher, so that check
cannot exercise the moving-teacher feedback loop. A true decoder batch-shape
difference in the first student update enters the EMA teacher and can alter the
next step's discrete target selection. A separate short B=1/B=2 pair would
duplicate work without bounding continued feedback. The actual main runs
provide the common-state first-step witness and the complete trajectory. Only
one extra B=1 first-step replay is retained to establish the local A/A floor.

## 4. Canonical Sample Schedule

A small Stage 1 dataset view pads the 3,450 source rows to the next multiple of
the effective global batch, 3,456. The six added slots map deterministically to
the first six source rows but receive distinct canonical slot IDs. They are
part of the sampler population before seeded shuffling, so B=1 and B=2 receive
the same 3,456-slot permutation for an epoch. `source_index` remains immutable
provenance; `canonical_slot_id` identifies a concrete occurrence in the padded
epoch and is unique within that epoch.

Source indices are zero-based. Before shuffling, view slots `0..3449` map to
the same source index and slots `3450..3455` map to source indices `0..5`.
For zero-based epoch `e`, a slot's run-unique ID is `e * 3456 + view_slot`.
The epoch permutation is exactly
`torch.randperm(3456, generator=torch.Generator('cpu').manual_seed(42 + e))`.
No data-loader worker RNG participates (`dataloader_num_workers=0`).

Rank `r` receives `permutation[r::4]`. For zero-based optimizer update `u`
within the epoch, each rank consumes its consecutive entries `4*u:4*u+4`.
B=1 presents those four entries as four one-row microbatches; B=2 presents the
same entries as two two-row microbatches. The zero-based upcoming
`optimizer_global_step = e * 216 + u` is the value used by stochastic keys.
The sampler calls `set_epoch(e)`, uses neither `drop_last` nor framework tail
padding, and checkpoints only at optimizer-step boundaries. The manifest and
trace record the permutation algorithm and PyTorch version, not only its seed.

The existing distributed batch sharding may assign samples within one global
update to different ranks or microbatch pairs. That assignment is a necessary
consequence of the true batch shape. The comparison contract is the canonical
ordered 16-sample global update, not identical rank-local grouping.

The training path validates before the first expensive forward that:

```text
padded_dataset_length % effective_global_batch == 0
rank dataloader length % gradient_accumulation_steps == 0
updates_per_epoch == 216
configured total updates == 648
```

No partial accumulation group, Accelerate tail recycling, or epoch-boundary
gradient carry is permitted. Unexpected sampler or dataloader lengths fail
closed. The slot/source identities for each canonical global update are
recorded only for the equivalence witness, not for ordinary production
logging.

## 5. Unified Per-Sample Objective

### 5.1 CE

Both B=1 and B>1 omit labels from the student model forward and use one
repository-owned CE implementation. It applies the causal shift, ignores
`-100`, computes the mean over valid tokens independently for each sample, and
then averages the sample losses. The same FP32, 1,024-token chunked reduction
is used for both batch sizes.

Each chunk computes an FP32 cross-entropy sum. Chunk sums are added from the
lowest token offset to the highest in FP32, divided by that row's exact valid
token count, and the resulting row scalars are added in row order and divided
by batch size. DeepSpeed/Trainer then divides each microbatch mean by GA, so
both groupings contribute exactly one quarter of each rank's four row losses
to an optimizer update. The tiny accumulated-gradient witness verifies that
actual framework path rather than inferring it from configuration.

This removes the current native-model-CE versus custom-CE branch. A sample
with zero valid shifted labels remains an error.

### 5.2 Alignment

Alignment is calculated separately for each batch row:

```text
sample_alignment = mean(loss over that sample's selected latent positions)
batch_alignment  = mean(sample_alignment over all batch rows)
```

It is never a flat mean over all latent positions in the microbatch. This
gives each source sample the same alignment weight as its B=1 loss regardless
of whether it has 8, 16, 24, 32, 40, or 48 selected latent rows. A row with no
selected latent position is always an error, matching the current valid-block
target contract. Every row therefore contributes one FP32 sample alignment to
the denominator, and the batch denominator is exactly batch size. Cosine,
row means, and the final batch mean are all calculated in FP32.

The total objective remains:

```text
loss = ce_weight * mean(sample_CE)
     + sim_weight * mean(sample_alignment)
```

## 6. Sample-Stable Stochastic Operations

Every stochastic training draw is keyed by:

```text
(base_seed, optimizer_global_step, canonical_slot_id, stable_site_name)
```

The key uses a stable digest rather than Python's process-randomized `hash()`.
It changes across optimizer steps and epochs because `optimizer_global_step`
changes, but it does not change when the same canonical update is grouped as
B=1 or B=2. A padded duplicate has an independent slot ID and therefore an
independent stochastic draw even when it maps to the same source row.

RNG schema `stage1-sample-rng-v1` encodes the key as canonical JSON UTF-8,
hashes it with SHA-256, and interprets the first eight digest bytes as an
unsigned little-endian integer masked to 63 bits. A fresh
`torch.Generator(device=<tensor device>)` receives that seed. Tensor hashes use
SHA-256 over canonical JSON `{dtype,shape}` followed by contiguous little-endian
CPU bytes; nonfinite values are rejected before hashing.

### 6.1 DoRA dropout

Repository-owned sample-stable dropout modules replace only the PEFT dropout
modules attached to Stage 1 decoder adapters. A stable module path is the site
name. Each row receives an independent device-local generator and inverted
dropout mask. For a row-local adapter input, the generator draws FP32 uniform
values over exactly its unpadded `[T,H]` extent, keeps values `>= p`, casts the
mask to the activation dtype, and scales by `1/(1-p)`. Padding in another row
cannot consume or shift its random stream.

The Stage 1 trainer holds the stochastic context around the complete
`training_step`, including backward recomputation. This is required because
gradient checkpointing re-executes adapter dropout after `compute_loss` has
returned. Missing, stale, or shape-incompatible stochastic context is an
error; silently falling back to global RNG is forbidden.

Stable site names are the fully-qualified PEFT module paths captured after
model composition and checked for uniqueness. The supported topology invokes
each site once per logical decoder forward. Checkpoint recomputation is marked
as a replay and regenerates the same mask without advancing an invocation
counter; an unmarked second logical invocation of one site is an error.

### 6.2 Alignment noise

Alignment target noise is selection-independent. For each sample, the stable
site `alignment_noise_bank` generates one FP32 standard-normal bank of shape
`[sum(all latent positions in metadata order), hidden_size]` before teacher
selection. Each latent position has one canonical bank row. After selection,
the trainer gathers the rows corresponding to selected latent positions and
adds `0.01 * noise` to the FP32 alignment targets. The full bank hash remains
an exact B=1/B=2 control even when later moving-EMA selection chooses different
subsets; gathered values are then model-derived observations.

Teacher inference remains in evaluation mode and receives no stochastic
training context.

### 6.3 Stochastic-module inventory

The Stage 1 startup check enumerates active training-mode stochastic modules.
Every enabled module must either be a registered sample-stable adapter dropout
site or be explicitly proven inactive by configuration. An unregistered active
dropout or stochastic layer is a launch error; the equivalence claim must not
assume that the adapter is the only RNG consumer.

The recorded allowlist covers both modules and functional/non-module RNG call
sites. Dataset preprocessing, target selection, and teacher inference are
deterministic and may not consume a training RNG stream. The manifest records
every allowed site and its schema; a runtime draw from any other site fails.

### 6.4 Checkpoint and resume

The seed derivation depends on restored `global_step` and the canonical slot
ID. No new mutable RNG stream needs to be checkpointed. Stage 1 metadata
records the sample-stable RNG schema and rejects a resume that changes it,
DoRA dropout, base seed, padded-dataset identity, or effective global batch.

## 7. Samplewise Frozen Vision and Teacher Targets

### 7.1 Disproved v1 premise

The v1 first-nail probe used the initial real 3B teacher on variable-length,
one- through six-block samples. Input IDs, attention masks, and position IDs
matched exactly, but the batched frozen visual encoder changed image
embeddings. The difference amplified through the teacher language model:

```text
worst teacher target relative L2       0.598
worst teacher target cosine            0.817
worst language hidden relative L2      0.744
worst language hidden cosine           0.711
selected candidate indices             different
```

Supplying the exact samplewise image embeddings reduced the true batched
decoder difference to relative L2 `0.0162` and cosine `0.99987`. Changing text
attention among eager, SDPA, and FlashAttention 2 did not remove the original
divergence. A proposed FP32 visual probe was rejected by the actual
FlashAttention visual kernel, which accepts only FP16/BF16.

### 7.2 Student visual boundary

The collator applies the processor independently to each
`PreparedStage1Example`, preserving B=1 preprocessing, then right-pads and
packs the row tensors for B=2. `Stage1SampleMetadata` owns zero-based half-open
ranges for that row's original and helper image grids, its unpadded token
length, and its latent spans. Source corruption, filtering, or truncation is a
preflight error rather than a silently skipped row.

The student's frozen visual encoder runs sequentially once per sample, never once per
latent block. Its row-local original-image features are injected into a
batched embedding tensor together with the row-local EMA latent targets. The
trainable language decoder then runs once on the true B=2 tensor. Text and
latent-token embedding gradients remain live; frozen visual execution remains
under `no_grad`. Row extraction preserves the exact unpadded sample inputs used
by B=1; B=2-only right-padding is introduced only while packing the decoder
inputs.

The implementation first creates the batched text/latent token embeddings with
autograd enabled, scatters BF16 samplewise visual outputs into original-image
token positions under `no_grad`, then scatters detached row-major teacher
targets into the selected latent positions. It supplies the original
`input_ids`, packed `inputs_embeds`, attention mask, image grids, and explicit
multimodal position IDs to one decoder call. The processor, visual module, and
teacher use the same BF16 autocast policy in both arms.

### 7.3 EMA teacher boundary

The EMA teacher constructs targets independently for each sample. For each
row it performs one language forward over that row's original image and text
and one helper-visual forward over all helper images owned by that row.
Multiple latent blocks reuse those outputs; no language or visual forward is
repeated per latent block. Padding from another B=2 row is removed before each
samplewise teacher call, so its token, image, and position inputs equal the B=1
call.

The independent row results are packed back into the existing row-major
`Stage1TeacherTargets` contract. Diagnostic selected candidate indices are
collected only when equivalence diagnostics are enabled and do not enter the
student interface or checkpoint artifact.

The preserved contract is the dataclass in `src/stage1/types.py`: one `[B,T]`
boolean latent mask, one `[N,H]` row-major target tensor, and one tuple of
per-block selected counts per sample. Candidate ownership and selection remain
the algorithms in `src/stage1/targets.py`, except equal similarity values are
tied deterministically by lower row-local candidate index. Every valid latent
block must select at least one row, and selected rows fill the block's latent
positions in order. The diagnostic selection margin is the sorted probability
of the last selected candidate minus the first excluded candidate; selecting
all candidates records a tagged `all_selected` value rather than infinity.

This deliberately gives up teacher and frozen-vision batch acceleration. It
is the cost of preserving the measured B=1 target semantics while retaining a
true B=2 trainable decoder. A future batched visual/teacher optimization is
outside this design unless it first proves the same row outputs.

All samplewise visual and teacher intermediates are scoped to one
`compute_loss` call and released after backward no longer needs the packed
student inputs. There is no cross-step feature cache or per-latent-block
recomputation.

## 8. Verification Layers

### 8.1 Focused unit tests

Tests must first fail on the current implementation and then cover:

- padded duplicates retain source provenance but have unique canonical slot
  IDs and independent stochastic keys;
- unequal token lengths: B=2 CE equals the mean of independent B=1 CE;
- unequal latent counts: B=2 alignment loss and gradients equal independent
  B=1 sample means when stochastic tensors are supplied identically;
- samplewise student visual embeddings and teacher targets equal independent
  B=1 calls before decoder packing;
- teacher selected counts and candidate indices equal independent B=1 calls;
- DoRA dropout masks are identical per valid sample position across B=1 and
  B=2, including unequal right padding and checkpoint recomputation;
- full alignment-noise banks are identical per sample across packed and
  separate calls on CUDA, and equal selections gather equal rows;
- 3,450 rows produce 3,456 canonical slots and exactly 216 complete updates
  for both configurations;
- canonical manifests reject every non-whitelisted semantic difference;
- resume recreates the next step's stochastic tensors from restored state.

### 8.2 Tiny end-to-end witness

An equal initial tiny Qwen/DoRA model trains for at least two optimizer steps
under B=1/GA=4 and B=2/GA=2. The witness compares canonical slot/source
identities, CE, alignment, total loss, accumulated trainable gradients,
optimizer state, scheduler state, student parameter deltas, and EMA parameter
deltas.

FP32 deterministic fixtures use `torch.testing.assert_close` with
`rtol=1e-6` and `atol=1e-7`. Exact integer, boolean, sample-identity,
selected-index, and step state uses equality.

### 8.3 Real 3B, four-GPU online control audit

The approved multi-node route starts the fresh B=1 and B=2 jobs sequentially
in the `llm_kd` container on `10.1.102.41` and the local node, using four GPUs
total: remote rank 0, local rank 1, two visible GPUs per node,
`MASTER_ADDR=10.1.102.41`, and both NCCL/GLOO interfaces `bond1.102`. It reuses
`run_stage1_training.sh` through `python -m src.multinode train-stage1`; no new
distributed launcher is introduced. The fresh B=1 reference runs uninterrupted
for all 648 optimizer steps and records the compact control trace plus the
requested tensor anchors. One
B=1 replay then executes only its first optimizer step from the same immutable
initial state and the same 648-step scheduler. It is the sole A/A run. Apart
from run/output identity and its bounded one-step exit, its resolved manifest
equals the B=1 reference. This measures real four-GPU kernel and collective
repeatability without another full training run.

The replay remains configured with the 648-step horizon; a diagnostic callback
exits immediately after completing and saving optimizer step 1. It does not set
`max_steps=1` and therefore cannot change warmup or scheduler construction.

Only after that A/A check passes does the fresh B=2 candidate start. Its first
optimizer step is the common-state A/B witness. On a pass, the same process
continues uninterrupted through step 648 while comparing compact exact-control
records online against the completed B=1 trace. There is no planned checkpoint
pause, resume, discarded warm-up, or separate B=1/B=2 short pair.

The launch-gating control audit compares:

- exact resolved manifests outside the explicit JSON-pointer whitelist;
- exact canonical global slot/source identities;
- exact per-slot hashes of unpadded input IDs, labels, attention masks,
  original/helper pixel tensors, and image grids at every step;
- exact per-slot stochastic keys, valid-position DoRA-mask hashes, and full
  canonical alignment-noise-bank hashes at every step;
- exact first-step selected counts and candidate indices;
- samplewise first-step student visual embeddings and initial EMA teacher
  target rows before decoder packing;
- exact first-step decoder position IDs at each slot's valid token positions;
- first-step flattened pre-clip trainable gradients and trainable student,
  EMA, and logical Adam first-/second-moment deltas;
- exact optimizer, scheduler, global-step, and EMA-update counters;
- measured wall time per optimizer step and peak allocated/reserved GPU memory.

The main B=1 first step and its A/A replay must match their floating tensors
with `rtol=1e-6` and `atol=1e-7`; exact identities and discrete states use
equality. Before the step, this includes base and trainable initialization,
initial EMA snapshot, empty optimizer state, scheduler state, data/sampler
state, and all four ranks' first canonical slots. If this repeatability
baseline fails, the A/B numerical difference cannot be attributed to
microbatch grouping and the B=2 main run is not launched.

Samplewise frozen-visual embeddings and first-step EMA teacher target rows in
B=1 versus B=2 execute through the same unpadded B=1-shaped calls from
identical state and must satisfy `rtol=1e-6` and `atol=1e-6`. First-step
selected counts and candidate indices must agree exactly.

For nonzero first-step trainable-decoder, gradient, update, EMA-delta, and
optimizer-state tensors, BF16 numerical acceptance requires both:

```text
symmetric relative L2 error <= 2.0e-2
cosine similarity           >= 0.9998
```

For two FP64 comparison vectors `a` and `b`, symmetric relative L2 is
`2*||a-b||_2/(||a||_2+||b||_2)` and cosine is
`dot(a,b)/(||a||_2*||b||_2)`. A category with both norms zero must match
exactly; a one-sided zero fails. Canonical vectors concatenate tensors in
lexicographically sorted fully-qualified parameter-name order, preserving
shape metadata. The independently gated categories are: accumulated pre-clip
gradient, student update delta, EMA update delta, Adam first moment, and Adam
second moment. Every category must pass both thresholds. Per-parameter errors
and the worst name are diagnostic only.

"Pre-clip" means after GA scaling, BF16 loss unscale, and distributed gradient
reduction but before the configured norm-1.0 clipping. ZeRO-1 logical Adam
moments and gradients are reconstructed on rank zero in canonical parameter
order solely for the checkpoint-tree comparison artifact. Student/EMA deltas
are measured against the immutable common initialization after the optimizer
and one EMA update have completed.

First-step scalar CE, alignment, and total losses must additionally have
absolute difference at most `1.0e-2`. Exact-zero tensors must agree exactly.
Maximum absolute error and the worst named tensor are always reported but are
not used alone because their scale is not comparable across parameters.
Backward accumulation and reduction order is part of the permitted GPU
batch-shape numerical boundary, not silently treated as exact arithmetic.

The numerical boundary is based on the observed `0.0162` relative-L2 true
decoder floor after exact samplewise visual inputs. It is not widened if the
gradient or update witness fails. These thresholds bound decoder batch-shape
effects at a common initial state; they do not permit a changed sample, initial
target selection, visual input, initial teacher target, update count, RNG draw,
or loss weighting.

### 8.4 EMA trajectory observation

After the common first update, student and EMA states are outcomes of the
paired executions. Later teacher target values or discrete selections may
therefore diverge even though every controllable input remains equal. Treating
that divergence as a launch failure would censor the trajectory difference
being measured.

For steps 2 through 648, both main runs record compact per-step traces of
selected counts and indices, selection-score margins, component losses,
gradient norms, student/EMA parameter norms, timing, and memory. Full tensors
for teacher targets, gradients, student/EMA deltas, and optimizer state are
compared at steps 2, 64, and 648 as observations using the same relative-L2 and
cosine metrics, but no post-step-1 numerical or discrete equality threshold is
a launch criterion. The report identifies the first target-selection
divergence and its margin when one occurs.

Parameter-shaped categories remain directly comparable at every anchor.
Teacher rows are joined by `(canonical_slot_id, latent_block_index,
candidate_index)`; metrics cover only matching rows and report the matched and
unmatched counts. Different selected shapes are therefore an observed
selection outcome, not an invalid tensor comparison.

Canonical slot/source identities, unpadded input hashes, stochastic keys,
valid-position dropout hashes, full alignment-noise-bank hashes, and
optimizer/scheduler/update counters remain exact controls through step 648.
The B=2 continuation checks these compact controls against the completed B=1
trace step by step and aborts on a mismatch. Gathered alignment-noise rows,
teacher decisions, and numerical trajectory values are observational after
the common first update.

The trace is canonical JSONL schema `stage1-equivalence-trace-v1`, one record
per optimizer step. Each record contains the manifest digest,
`rng_global_step` in `0..647`, `optimizer_step_completed` in `1..648`, ordered
per-rank slot/source IDs, SHA-256 control hashes, exact counters, teacher
selected indices/counts/margins, finite scalar summaries, timing, and memory.
Rank zero writes only after gathering all four ranks; duplicate or missing
steps fail. Model-bearing anchor tensors use `torch.save` dictionaries under
`checkpoints/<run-id>/equivalence/`, keyed by the same canonical category and
fully-qualified parameter names. Their derived metrics alone enter `results/`.

The one-step A/A replay establishes local repeatability only. This design does
not run a third 648-step B=1 replay or multiple paired seeds, so it cannot
separate amplified long-horizon hardware/kernel nondeterminism from a general
microbatch-grouping effect. Final trajectory and accuracy differences are
reported as one controlled paired observation, not as an estimated causal
effect or a generalizable performance delta. That limitation is an explicit
cost waiver, not evidence that long-horizon A/A variance is zero.

## 9. Launch Gate and Artifacts

The B=1 reference may start only after the focused tests and tiny witness pass.
The B=2 candidate may start only after the B=1 run completes and its one-step
A/A replay passes. After the B=2 first step, a control or numerical threshold
failure stops that run and is reported without retrying at looser thresholds.
On a pass, B=2 continues uninterrupted; later EMA trajectory divergence is
reported but does not fail the run, while any exact controllable-state mismatch
does. Both completed models are then evaluated with one resolved official
evaluation manifest and the same fixed test set. Accuracy and exact
per-example predictions are reported side by side; full-run accuracy equality
is an observed comparison, not a launch threshold, because permitted GPU
numerical differences can compound.

Official evaluation reuses `run_multinode_evaluation.sh` with four
deterministic source-index shards over the fixed TEST digest and generation
values in Section 2. Prepared inference models live only under
`checkpoints/evaluation-staging/<run-id>/`. The two evaluation manifests may
differ only in model/checkpoint identity and run/output identity. The report
joins predictions by zero-based TEST `source_index`, publishes each model's
exact-correct count over 400 and accuracy, and lists every per-example
agreement or disagreement.

Artifacts follow repository policy:

```text
checkpoints/<run-id>/   model, optimizer, scheduler, EMA, RNG state
results/<run-id>/       equivalence evidence and comparisons
logs/<run-id>/          execution logs
```

Anchor visual embeddings, teacher targets, gradients, optimizer state,
student deltas, EMA deltas, or other model-derived comparison tensors are
model-bearing artifacts and remain under the corresponding
`checkpoints/<run-id>/equivalence/` directory. `results/` contains only their
derived metrics and discrete traces.

No model-bearing file is written elsewhere in the repository. Verification
uses fresh run IDs and never overwrites the completed 36% evaluation or prior
training artifacts.

## 10. Expected Code Boundaries

Expected modifications are limited to the existing Stage 1 path:

```text
src/stage1/data.py           padded view, source/slot identity, visual ownership
src/stage1/trainer.py        unified loss, samplewise vision/teacher, RNG context
src/stage1/targets.py        samplewise teacher and deterministic selection
src/stage1/peft.py           adapter-dropout replacement and stable site names
src/stage1/types.py          equivalence metadata/diagnostic contracts
src/stage1/equivalence.py    manifest, sample RNG, hashes, trace comparison
src/stage1/checkpointing.py  RNG schema and resume compatibility metadata
src/stage1/composition.py    schedule construction and launch validation
src/stage1/training_evidence.py  control/trajectory evidence
src/main.py                  bounded equivalence-diagnostic arguments
src/multinode.py             reuse of the existing launch/evaluation route
tests/stage1/...             focused unit and tiny integration witnesses
```

A small, direct comparison script may be added under `tests/stage1/` or the
existing Stage 1 diagnostic launcher. It must reuse the existing multi-node
launcher rather than introduce another orchestration subsystem.

No Stage 2 objective, evaluation metric, generic Qwen generation path, or
DeepSpeed implementation is changed.

## 11. Error Handling

Fail before or during candidate training for:

- non-divisible canonical epoch or rank-local accumulation schedule;
- missing or duplicate canonical slot IDs within a microbatch;
- a non-whitelisted resolved-manifest difference;
- an unregistered active training-mode stochastic module;
- missing stochastic context while sample-stable dropout is active;
- mismatch between checkpoint RNG schema and requested run;
- an A/A repeatability failure;
- a B=1/B=2 first-step target-selection or numerical threshold failure;
- nonfinite loss, gradient, parameter delta, EMA delta, or optimizer moment;
- any controllable exact-state mismatch.

After a runtime failure, retain the log, identify one root cause, add one
causal regression test, and retry with a fresh run ID. Broad retry or recovery
infrastructure is outside this design.

## 12. Non-Goals

- bitwise B=1/B=2 GPU equality;
- serial trainable decoder forwards hidden behind `B=2`;
- batched frozen visual or teacher execution without an independent
  equivalence witness;
- changing DoRA rank, alpha, dropout, target modules, or learning rate;
- removing ILVR alignment noise;
- changing EMA tau or teacher-update semantics;
- requiring identical moving-teacher trajectories after the common first
  update;
- estimating a general batch-grouping effect across full-run A/A replicates or
  multiple seeds;
- Stage 2 batch-size work;
- a new benchmark framework, retry framework, or experiment database;
- starting B=2 before the completed B=1 trace and one-step A/A check pass.

## 13. Completion Claim

Completion means the repository can produce fresh evidence that B=1/GA=4 and
true B=2/GA=2 use identical manifests outside the approved batch/output
whitelist and receive the same controllable sample inputs, stochastic draws,
objectives, schedules, and update boundaries through the online audit. Their
common-state first update differs only within the bounded GPU
forward/backward witness and against a measured A/A repeatability baseline.
Completion also includes observing the resulting EMA trajectory through all
648 steps, identifying any first teacher-selection divergence, and reporting
the paired official evaluations as a single paired observation. It does not
claim identical post-update teachers, bitwise-identical checkpoints, identical
accuracy, or a causal/generalizable endpoint effect without a full-horizon
A/A or multi-seed baseline.
