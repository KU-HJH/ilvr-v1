# Stage 1 True-Batch Equivalence Design

**Date:** 2026-09-01  
**Status:** Design proposal v4, frozen for critique
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

The trainable language decoder remains a true B=2 operation. Frozen visual
features and EMA-teacher targets are computed sample by sample before that
decoder call because an actual 3B probe disproved their batch invariance. The
only permitted differences are floating-point effects caused by executing the
same decoder mathematics through different true batch shapes on GPU. Bitwise
decoder equality is not a goal because it would require serial B=1 decoder
forwards and remove the requested batching benefit.

## 2. Fixed Experiment Contract

The controlled comparison is:

```text
model                         Qwen2.5-VL-3B-Instruct
method                        DoRA
DeepSpeed                     ZeRO-1
world size                    4
seed                          42
epochs                        3
dataset                       data/TRAIN.jsonl, 3,450 source rows
canonical epoch slots         3,456
effective global batch        16
reference                     B=1, GA=4
candidate                     B=2, GA=2
optimizer steps per epoch     216
total optimizer steps         648
EMA tau                       0.999
learning rate                 1e-5
DoRA dropout                  0.05
alignment target noise        0.01
maximum sequence length       32,768
attention                     FlashAttention 2
precision                     BF16 training, FP32 loss reductions
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

### 3.4 Rejected: a separate one- or two-step real-model gate

At the first step both paths use the same initial EMA teacher, so that check
cannot exercise the moving-teacher feedback loop. A true decoder batch-shape
difference in the first student update enters the EMA teacher and can alter the
next step's discrete target selection. Two steps exercise only the first such
opportunity and do not bound continued feedback. A separate short job would
also duplicate work. The real witness is therefore the first 64 steps of the
actual paired runs, paused once for comparison and resumed only on a pass.

## 4. Canonical Sample Schedule

A small Stage 1 dataset view pads the 3,450 source rows to the next multiple of
the effective global batch, 3,456. The six added slots map deterministically to
the first six source rows but receive distinct canonical slot IDs. They are
part of the sampler population before seeded shuffling, so B=1 and B=2 receive
the same 3,456-slot permutation for an epoch. `source_index` remains immutable
provenance; `canonical_slot_id` identifies a concrete occurrence in the padded
epoch and is unique within that epoch.

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

This removes the current native-model-CE versus custom-CE branch. A sample
with zero valid shifted labels remains an error.

### 5.2 Alignment

Alignment is calculated separately for each batch row:

```text
sample_alignment = mean(loss over that sample's selected latent positions)
batch_alignment  = mean(sample_alignment over valid samples)
```

It is never a flat mean over all latent positions in the microbatch. This
gives each source sample the same alignment weight as its B=1 loss regardless
of whether it has 8, 16, 24, 32, 40, or 48 selected latent rows. A row with no
selected latent position contributes a defined zero only when the existing
target contract permits such a row; otherwise the existing target validation
continues to fail.

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

### 6.1 DoRA dropout

Repository-owned sample-stable dropout modules replace only the PEFT dropout
modules attached to Stage 1 decoder adapters. A stable module path is the site
name. Each row receives an independent device-local generator and inverted
dropout mask. Padding in another row cannot consume or shift its random stream.

The Stage 1 trainer holds the stochastic context around the complete
`training_step`, including backward recomputation. This is required because
gradient checkpointing re-executes adapter dropout after `compute_loss` has
returned. Missing, stale, or shape-incompatible stochastic context is an
error; silently falling back to global RNG is forbidden.

### 6.2 Alignment noise

Alignment target noise is generated row by row with the same key scheme and a
fixed alignment site name. The draw shape is exactly that sample's selected
latent matrix, so concatenating another sample cannot alter its values.

Teacher inference remains in evaluation mode and receives no stochastic
training context.

### 6.3 Stochastic-module inventory

The Stage 1 startup check enumerates active training-mode stochastic modules.
Every enabled module must either be a registered sample-stable adapter dropout
site or be explicitly proven inactive by configuration. An unregistered active
dropout or stochastic layer is a launch error; the equivalence claim must not
assume that the adapter is the only RNG consumer.

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

The student's frozen visual encoder runs once per sample, never once per
latent block. Its row-local original-image features are injected into a
batched embedding tensor together with the row-local EMA latent targets. The
trainable language decoder then runs once on the true B=2 tensor. Text and
latent-token embedding gradients remain live; frozen visual execution remains
under `no_grad`. Row extraction preserves the exact unpadded sample inputs used
by B=1; B=2-only right-padding is introduced only while packing the decoder
inputs.

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

This deliberately gives up teacher and frozen-vision batch acceleration. It
is the cost of preserving the measured B=1 target semantics while retaining a
true B=2 trainable decoder. A future batched visual/teacher optimization is
outside this design unless it first proves the same row outputs.

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
- alignment-noise tensors are identical per sample across packed and separate
  calls on CUDA;
- 3,450 rows produce 3,456 canonical slots and exactly 216 complete updates
  for both configurations;
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

### 8.3 Real 3B, four-GPU 64-step staged witness

The approved multi-node route starts the fresh B=1 and B=2 jobs sequentially
in the `llm_kd` container on `10.1.102.41` and the local node, using four GPUs
total. Both are configured from the outset for the final 648 optimizer steps,
including the same 648-step scheduler. A diagnostic stop callback saves and
pauses each run after optimizer step 64 without changing `max_steps` or the
scheduler horizon. This is about one tenth of the paired training compute, and
the successful work is not discarded: both runs resume from those exact
checkpoints after the comparison passes.

The witness compares:

- exact resolved manifests outside the explicit four-field whitelist;
- exact canonical global slot/source identities;
- exact per-slot hashes of unpadded input IDs, labels, attention masks,
  original/helper pixel tensors, and image grids at every step;
- exact per-slot stochastic keys and valid-position DoRA-mask/alignment-noise
  hashes at every step;
- exact selected counts and candidate indices at every step from 1 through 64;
- finite per-component loss traces at every step;
- samplewise student visual embeddings and EMA teacher target rows at anchor
  steps 1, 2, and 64 before decoder packing;
- exact decoder position IDs at the same anchor steps;
- flattened pre-clip trainable gradients at anchor steps 1, 2, and 64;
- trainable student, EMA, and logical Adam first-/second-moment deltas at the
  same anchor steps;
- exact optimizer, scheduler, global-step, and EMA-update counters;
- measured wall time per optimizer step and peak allocated/reserved GPU memory.

Samplewise frozen-visual embeddings and first-step EMA teacher target rows
execute through the same unpadded B=1-shaped calls from identical state and
must satisfy `rtol=1e-6` and `atol=1e-6`. Later target rows may inherit only
the already-bounded student/EMA update difference and use the decoder numerical
boundary below. Selected counts and candidate indices must agree exactly at
all 64 steps.

For nonzero later target, trainable-decoder, gradient, update, EMA-delta, and
optimizer-state tensors at each recorded anchor, BF16 numerical acceptance
requires both:

```text
symmetric relative L2 error <= 2.0e-2
cosine similarity           >= 0.9998
```

Scalar CE, alignment, and total losses must additionally have absolute
difference at most `1.0e-2` at every step. Exact-zero tensors must agree
exactly. Maximum absolute error and the worst named tensor are always reported
but are not used alone because their scale is not comparable across
parameters. Backward accumulation and reduction order is part of the permitted
GPU batch-shape numerical boundary, not silently treated as exact arithmetic.

The numerical boundary is based on the observed `0.0162` relative-L2 true
decoder floor after exact samplewise visual inputs. It is not widened if the
gradient or update witness fails. These thresholds bound decoder batch-shape
effects; they do not permit a changed sample, target selection, visual input,
teacher target, update count, RNG draw, or loss weighting.

## 9. Launch Gate and Artifacts

The paired runs may proceed beyond step 64 only after all focused tests, the
tiny witness, and the staged real comparison pass. A failure is a result, not
a reason to retry with looser thresholds; neither run resumes. On a pass, the
same B=1/GA=4 and B=2/GA=2 runs resume from their step-64 checkpoints and finish
three epochs. Both are then evaluated with one resolved official evaluation
manifest and the same fixed test set. Accuracy and exact per-example
predictions are reported side by side; full-run accuracy equality is an
observed comparison, not a launch threshold, because permitted GPU numerical
differences can compound.

Artifacts follow repository policy:

```text
checkpoints/<run-id>/   model, optimizer, scheduler, EMA, RNG state
results/<run-id>/       equivalence evidence and comparisons
logs/<run-id>/          execution logs
```

Anchor gradient, optimizer, student, EMA, or other model-derived comparison
tensors are model-bearing artifacts and remain under the corresponding
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
src/stage1/peft.py           adapter-dropout replacement and stable site names
src/stage1/types.py          equivalence metadata/diagnostic contracts
src/stage1/checkpointing.py  RNG schema and resume compatibility metadata
src/stage1/composition.py    schedule construction and launch validation
src/stage1/training_evidence.py  staged witness evidence
src/main.py                  bounded diagnostic-stop argument
src/multinode.py             reuse of the existing launch/resume route
tests/stage1/...             focused unit and tiny integration witnesses
```

A small, direct comparison script may be added under `tests/stage1/` or the
existing Stage 1 diagnostic launcher. It must reuse the existing multi-node
launcher rather than introduce another orchestration subsystem.

No Stage 2 objective, evaluation metric, generic Qwen generation path, or
DeepSpeed implementation is changed.

## 11. Error Handling

Fail before full training for:

- non-divisible canonical epoch or rank-local accumulation schedule;
- missing or duplicate canonical slot IDs within a microbatch;
- a non-whitelisted resolved-manifest difference;
- an unregistered active training-mode stochastic module;
- missing stochastic context while sample-stable dropout is active;
- mismatch between checkpoint RNG schema and requested run;
- any discrete teacher target-selection difference in the real witness;
- nonfinite loss, gradient, parameter delta, EMA delta, or optimizer moment;
- any exact-state mismatch or numerical threshold failure.

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
- Stage 2 batch-size work;
- a new benchmark framework, retry framework, or experiment database;
- proceeding past step 64 before the staged equivalence gate passes.

## 13. Completion Claim

Completion means the repository can produce fresh evidence that B=1/GA=4 and
true B=2/GA=2 use identical manifests outside the approved batch/output
whitelist and receive the same samplewise frozen visual inputs, teacher
targets, stochastic draws, objectives, schedules, and update boundaries for
64 real 3B optimizer updates, including the moving-EMA feedback trajectory.
The trainable decoder remains truly batched and may differ only within the
bounded GPU forward/backward witness. Completion also includes resuming the
same paired runs through three epochs and their side-by-side official
evaluation; it does not claim bitwise-identical checkpoints or identical
accuracy.
