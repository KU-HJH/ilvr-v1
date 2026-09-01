# Stage 1 True-Batch Equivalence Design

**Date:** 2026-09-01  
**Status:** Design proposal v1, frozen for critique  
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

The only permitted differences are floating-point effects caused by executing
the same per-sample mathematics through different true batch shapes on GPU.
Bitwise equality is not a goal because it would require serial B=1 student
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
one-step witnesses start from the same immutable initial model and tuning
state.

## 3. Considered Approaches

### 3.1 Selected: true B=2 with sample-stable semantics

Keep the student batched and make stochastic draws, objectives, and the global
sample schedule independent of microbatch grouping. This retains the intended
throughput advantage and makes the remaining numerical difference observable.

### 3.2 Rejected: disable dropout and alignment noise

This is simpler but changes the approved DoRA and ILVR objectives. It would
prove equivalence for a different experiment rather than the requested one.

### 3.3 Rejected: serialize each B=2 row as a B=1 forward

This provides strict gradient equality but is not true batching and forfeits
the memory and throughput behavior under investigation.

## 4. Canonical Sample Schedule

A small Stage 1 dataset view pads the 3,450 source rows to the next multiple of
the effective global batch, 3,456. The six added slots map deterministically to
the first six dataset-view indices. They are part of the sampler population
before seeded shuffling, so B=1 and B=2 receive the same 3,456-item permutation
for an epoch.

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
closed. The source indices for each canonical global update are recorded only
for the equivalence witness, not for ordinary production logging.

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
(base_seed, optimizer_global_step, source_index, stable_site_name)
```

The key uses a stable digest rather than Python's process-randomized `hash()`.
It changes across optimizer steps and epochs because `optimizer_global_step`
changes, but it does not change when the same canonical update is grouped as
B=1 or B=2.

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

### 6.3 Checkpoint and resume

The seed derivation depends on restored `global_step` and the stable source row
index. No new mutable RNG stream needs to be checkpointed. Stage 1 metadata
records the sample-stable RNG schema and rejects a resume that changes it,
DoRA dropout, base seed, or effective global batch.

## 7. Teacher Targets and Numerical Boundary

The EMA teacher keeps one batched language forward and one batched helper
visual forward. Sample ownership, masks, and selected counts remain isolated
by `Stage1BatchMetadata`.

True B=2 teacher kernels may produce small BF16 differences from separate B=1
teacher calls. The one-step witness therefore checks, per source sample:

- selected latent counts exactly;
- diagnostic selected candidate indices exactly;
- latent target values within the numerical acceptance boundary.

Diagnostic candidate indices are collected only when equivalence diagnostics
are enabled and do not enter the student interface or checkpoint artifact.
If a discrete target selection differs, the witness fails even if aggregate
losses look close. The implementation does not hide this difference by
loosening a floating-point tolerance. The full three-epoch run is then blocked
and any change to teacher execution requires a revised design approval.

## 8. Verification Layers

### 8.1 Focused unit tests

Tests must first fail on the current implementation and then cover:

- unequal token lengths: B=2 CE equals the mean of independent B=1 CE;
- unequal latent counts: B=2 alignment loss and gradients equal independent
  B=1 sample means when stochastic tensors are supplied identically;
- DoRA dropout masks are identical per valid sample position across B=1 and
  B=2, including unequal right padding and checkpoint recomputation;
- alignment-noise tensors are identical per sample across packed and separate
  calls on CUDA;
- 3,450 rows produce 3,456 canonical slots and exactly 216 complete updates
  for both configurations;
- resume recreates the next step's stochastic tensors from restored state.

### 8.2 Tiny end-to-end witness

An equal initial tiny Qwen/DoRA model trains for at least two optimizer steps
under B=1/GA=4 and B=2/GA=2. The witness compares canonical source indices,
CE, alignment, total loss, accumulated trainable gradients, optimizer state,
scheduler state, student parameter deltas, and EMA parameter deltas.

FP32 deterministic fixtures use `torch.testing.assert_close` with
`rtol=1e-6` and `atol=1e-7`. Exact integer, boolean, sample-identity,
selected-index, and step state uses equality.

### 8.3 Real 3B, four-GPU one-step witness

The approved multi-node route runs two fresh, sequential one-step jobs in the
`llm_kd` container on `10.1.102.41` and the local node, using four GPUs total.
Both jobs start from the same base model and seed and use distinct run IDs.

The witness compares:

- exact canonical global source indices;
- exact selected counts and candidate indices;
- exact optimizer, scheduler, global-step, and EMA-update counters;
- finite per-component losses;
- flattened pre-clip trainable gradients;
- trainable student update deltas;
- EMA update deltas;
- logical Adam first- and second-moment state.

For nonzero floating tensors, BF16 numerical acceptance requires both:

```text
symmetric relative L2 error <= 1.0e-2
cosine similarity           >= 0.9999
```

Scalar CE, alignment, and total losses must additionally have absolute
difference at most `1.0e-2`. Exact-zero tensors must agree exactly. Maximum
absolute error and the worst named tensor are always reported but are not used
alone because their scale is not comparable across parameters.

These thresholds bound BF16 batch-shape effects; they do not permit a changed
sample, target selection, update count, RNG draw, or loss weighting.

## 9. Launch Gate and Artifacts

The full three-epoch B=2 run may start only after all focused tests, the tiny
witness, and both real one-step jobs pass. A failed real witness is a result,
not a reason to retry with looser thresholds.

Artifacts follow repository policy:

```text
checkpoints/<run-id>/   model, optimizer, scheduler, EMA, RNG state
results/<run-id>/       equivalence evidence and comparisons
logs/<run-id>/          execution logs
```

No model-bearing file is written elsewhere in the repository. Verification
uses fresh run IDs and never overwrites the completed 36% evaluation or prior
training artifacts.

## 10. Expected Code Boundaries

Expected modifications are limited to the existing Stage 1 path:

```text
src/stage1/data.py           canonical padded dataset view and source identity
src/stage1/trainer.py        unified loss, per-sample alignment, RNG context
src/stage1/peft.py           adapter-dropout replacement and stable site names
src/stage1/types.py          equivalence metadata/diagnostic contracts
src/stage1/checkpointing.py  RNG schema and resume compatibility metadata
src/stage1/composition.py    schedule construction and launch validation
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
- missing or duplicate stable source indices within a microbatch;
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
- serial student forwards hidden behind `B=2`;
- changing DoRA rank, alpha, dropout, target modules, or learning rate;
- removing ILVR alignment noise;
- changing EMA tau or teacher-update semantics;
- Stage 2 batch-size work;
- a new benchmark framework, retry framework, or experiment database;
- starting a three-epoch run before the one-step equivalence gate passes.

## 13. Completion Claim

Completion means the repository can produce fresh evidence that B=1/GA=4 and
true B=2/GA=2 receive the same controllable training semantics for one real
3B optimizer update, with only bounded BF16 batch-kernel differences. It does
not mean the two three-epoch checkpoints will be bitwise identical or achieve
identical evaluation accuracy.
