# Stage 1 True-Batch Controlled Comparison Design

**Date:** 2026-09-01  
**Status:** Design commitment v20, approved 2026-09-01
**Repository:** `XD111ds/ILVR`  
**Supersedes:** only the batch-equivalence portions of
`2026-08-26-ilvr-stage1-dora-batched-training-design.md`

## 1. Goal

Make Stage 1 `B=2, gradient_accumulation=2` match every controllable training
semantic of a fresh `B=1, gradient_accumulation=4` reference at world size four
and effective global batch 16 while retaining a true two-row student forward.
Then measure the common-state forward/update perturbation alongside two
observed same-arm repeat discrepancies and observe one resulting paired
trajectory. Every exogenous training semantic and control entering that
computation must agree between the two paths:

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
decoder call because an actual 3B probe disproved their batch invariance. At
the common initial state, numerical differences from same-shape samplewise
visual/teacher calls plus decoder forward, backward, and reduction execution
are accepted only against fixed engineering thresholds and reported beside
the two observed same-arm discrepancies.
Gradients and updates are computed outcomes, not controllable inputs. After
the first update, derived state/teacher differences are also outcomes and are
not causally attributed by this one pair. This is not a claim of identical
trajectories or strict batch equivalence. Bitwise decoder equality is not a
goal because it would require serial B=1 decoder forwards and remove the
requested batching benefit.

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
CE weight / sim weight        1.0 / 1.0
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
the dataset, source JSONL, test JSONL, DeepSpeed JSON, and runtime source tree
are represented by SHA-256 digests. Trainable initialization is represented by
the ordered fully-qualified parameter name, dtype, shape, and tensor SHA-256.

Code identity is `/code/git_commit` plus `/code/runtime_source_digest`. The
latter hashes the sorted repository-relative path and bytes of every runtime
Python/shell source under `src/`, `eval.py`, `run_stage1_training.sh`,
`run_multinode_evaluation.sh`, and the loaded Qwen model source. Tracked
changes to those files relative to the recorded commit are a launch error;
unrelated untracked data, caches, and documentation do not invalidate the run.
The manifest also records the resolved Python, PyTorch, Transformers, PEFT,
DeepSpeed, CUDA, and FlashAttention versions.

The manifest schema is a closed top-level object with exactly these owned
sections:

```text
schema_version   schema string
model            model/processor file identities and model config
code             commit, runtime-source digest, resolved software versions
data             TRAIN/TEST/image identities and canonical padded view
batch            microbatch, GA, effective batch, rank-local grouping
schedule         epochs, permutation rule, steps, logging/checkpoint cadence
tuning           DoRA topology and loss/target hyperparameters
optimizer        AdamW and clipping fields
scheduler        type, warmup, total horizon
ema              tau, initialization, update rule
precision        autocast, loss reduction, attention, checkpointing
stochastic       schemas, site allowlist, dropout/noise specifications
deepspeed        source digest and non-batch resolved settings
evaluation       dataset, sharding, generation, scorer identity
diagnostics      bounded step and evidence schema
artifacts        run ID and checkpoint/result/log destinations
initialization   ordered trainable/EMA tensor identities
```

Each schema version defines the exact required child keys in code; missing or
additional keys fail validation rather than being ignored.

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

Every manifest contains `/diagnostics/bounded_optimizer_steps`: its value is
JSON `null` for a main run and integer `1` for a bounded replay. A replay and
its same-arm main manifest may differ only at that pointer and the four
`/artifacts/` pointers above; its three `/batch/` values must be identical. A
B=1/B=2 main comparison and the initial B=1/B=2 replay comparison may differ
only at the seven pointers in the preceding list. No cross-arm replay/main
manifest comparison is used as a gate.
Before any normalization, `/batch/rank_local_microbatch_grouping` must equal
the arm-specific mapping in Section 4; normalization permits the approved
difference but never permits an arbitrary value.

The immutable source DeepSpeed JSON is represented only by
`/deepspeed/source_digest`. Generated DeepSpeed `auto` values are resolved
before comparison into the three `/batch/` fields above and are not duplicated
under `/deepspeed`; all other resolved DeepSpeed settings are equal semantic
fields under `/deepspeed/settings`. Paths are stored as repository-relative
logical identities where possible; host-specific absolute mirrors are runtime
evidence and cannot mask a semantic digest difference.

Comparison parses JSON with duplicate-key rejection and validates that every
schema field and every allowed pointer is present; absent and `null` are never
equivalent. It deep-copies each manifest, replaces the value at every pointer
allowed for that comparison with the JSON string
`"<approved-difference>"`, serializes with the canonical JSON rules above, and
requires byte equality. The SHA-256 of those normalized bytes is the
normalized digest for that named comparison profile; the unmodified manifest
digest is retained only as run identity. Profile `cross_arm_v1` normalizes the
seven batch/artifact pointers, while `same_arm_v1` normalizes the four artifact
pointers plus `/diagnostics/bounded_optimizer_steps`. Every run records both
digests in `normalized_manifest_digests`; a comparator uses only its named
profile and rejects an unknown profile.

## 3. Considered Approaches

### 3.1 Selected: true B=2 decoder with sample-stable semantics

Keep the trainable student decoder batched and make frozen visual inputs,
teacher targets, stochastic draws, objectives, and the global sample schedule
independent of microbatch grouping. This retains decoder batching while making
the remaining numerical difference observable.

### 3.2 Rejected: disable dropout and alignment noise

This is simpler but changes the approved DoRA and ILVR objectives. It would
characterize a different experiment rather than the requested one.

### 3.3 Rejected: serialize each B=2 row as a B=1 forward

This provides strict gradient equality but is not true batching and forfeits
the memory and throughput behavior under investigation.

### 3.4 Rejected: stop after the paired first-step jobs

At the first step both paths use the same initial EMA teacher, so that check
cannot exercise the moving-teacher feedback loop. A true decoder batch-shape
difference in the first student update enters the EMA teacher and can alter the
next step's discrete target selection. The two bounded jobs are therefore only
the cheap common-state A/B launch gate, not the answer. The actual main runs
provide the paired trajectory, and each main's first step against its same-arm
bounded job records one repeat discrepancy for that arm.

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
`permutation_offset` is the zero-based index `0..3455` into that epoch's
permutation result, distinct from both the permuted view-slot value and the
run-unique canonical slot ID.

Rank `r` receives `permutation[r::4]`. For zero-based optimizer update `u`
within the epoch, each rank consumes its consecutive entries `4*u:4*u+4`.
B=1 presents those four entries as four one-row microbatches; B=2 presents the
same entries as two two-row microbatches. The zero-based upcoming
`optimizer_global_step = e * 216 + u` is the value used by stochastic keys.
The sampler calls `set_epoch(e)`, uses neither `drop_last` nor framework tail
padding, and checkpoints only at optimizer-step boundaries. The manifest and
trace record the permutation algorithm and PyTorch version, not only its seed.

The rank assignment above is exact across arms: a permutation offset assigned
to rank `r` may not move to another rank. Only its rank-local microbatch
ordinal changes—four singleton ordinals for B=1 versus two paired ordinals for
B=2. The comparison contract therefore fixes both the canonical ordered
16-sample global update and slot-to-rank placement, while permitting only the
approved within-rank microbatch grouping difference.

The arm-specific row order is closed, not merely descriptive. For rank `r` and
rank-local sample ordinal `j=0..3`, the global permutation offset is
`16*u + 4*j + r`. B=1 assigns it to `(microbatch=j, row=0)`; B=2 assigns it to
`(microbatch=j//2, row=j%2)`. Startup validates this mapping for all 648
updates, and every trace record carries and validates the same ordinals. A
different pair or row order is a schedule failure, even though the two approved
arm mappings intentionally differ from each other.

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

`ce_weight` and `sim_weight` are the exact CLI, trainer, checkpoint, and
manifest keys. “Alignment weight” is only a prose description of
`sim_weight`; no second alias or compatibility mapping exists.

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

The replacement keeps PEFT's `dropout(x)` call signature. The trainer overrides
`training_step`, derives the ordered row tuple
`(canonical_slot_id, valid_token_length)` from that microbatch, and installs a
`Stage1StochasticContext` containing those rows, the upcoming optimizer step,
and execution phase through a `ContextVar` before calling the normal forward
and backward path. In the supported Qwen linear DoRA
topology, every targeted PEFT layer passes a dense `[B,T,H]` activation; the
replacement validates rank three, `B` against the context, `T` against every
valid length, and its fully-qualified site name. It constructs an identity
mask of `ones_like(x)`, fills only each row's `[:valid_token_length, :]` slice
from that row's keyed generator, and returns `x * mask`. Any other activation
layout is a launch error rather than an inferred row mapping.

The Stage 1 trainer holds the stochastic context around the complete
`training_step`, including backward recomputation. Non-reentrant gradient
checkpointing uses its `context_fn` pair to mark original-forward versus
recompute phase. The original phase and its one recompute may invoke each
stable site once and deterministically regenerate the same mask from the same
key; a second invocation in either phase is an error. Missing, stale, or
shape-incompatible stochastic context is also an error; silently falling back
to global RNG is forbidden.

Composition installs that pair through:

```python
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={
        "use_reentrant": False,
        "context_fn": stage1_checkpoint_context_fn,
    }
)
```

A startup assertion inspects the model's resolved checkpoint function before
training. The `training_step`
context owns two maps keyed by stable site name, one for original-forward and
one for recompute invocation counts. Both maps start empty at each microbatch
entry and are cleared in the same `finally` block that removes the
`ContextVar`. Every site must have original count exactly one; recompute count
may be zero or one because non-reentrant early-stop can omit an unneeded site,
but may never exceed one.

At microbatch entry, `training_step` sets one parent `ContextVar` token whose
value owns immutable row/step data, both invocation maps, and phase
`"outside_checkpoint"`. The checkpoint `context_fn` returns two nested context
managers: forward temporarily sets phase `"original"`, recompute temporarily
sets `"recompute"`, and each restores the immediately previous phase on exit
without replacing or resetting the parent token. The `training_step` finally
block verifies counts, clears maps, and resets exactly that parent token only
after backward returns. Any adapter call in phase `"outside_checkpoint"` is
counted in the original map; any call after parent reset raises missing-context.

Right-padding positions receive the multiplicative identity mask `1`, consume
no RNG draw, and are excluded from valid-mask hashes and every loss. Thus the
valid `[T,H]` prefix of a sample is unchanged by another row's padding.

Stable site names are the fully-qualified PEFT module paths captured after
model composition and checked for uniqueness. The supported topology invokes
each site once per logical decoder forward. Recompute regenerates the same mask
without changing the logical invocation count.

### 6.2 Alignment noise

Alignment target noise is selection-independent. For each sample, the stable
site `alignment_noise_bank` generates one FP32 standard-normal bank of shape
`[sum(all latent positions in metadata order), hidden_size]` before teacher
selection. Bank rows are ordered first by metadata latent-block order and then
by each block's destination `latent_positions` order. When a block selects
`k` teacher candidates, they fill the first `k` destination positions and the
trainer gathers those same `k` bank rows. It adds `0.01 * noise` only to the
detached FP32 target used by cosine alignment loss. Clean EMA teacher values,
not noisy values, are scattered into student `inputs_embeds`; noise never
changes decoder inputs. The full bank hash remains an exact B=1/B=2 control
even when later moving-EMA selection chooses different subsets; gathered
values are then model-derived observations.

Teacher inference remains in evaluation mode and receives no stochastic
training context.

### 6.3 Stochastic-module inventory

The Stage 1 startup check enumerates every training-mode `nn.Dropout` module.
Each enabled one must be a registered sample-stable PEFT adapter site; any
other enabled dropout module is a launch error. The manifest records the exact
site list, paths, probability, and RNG schema. The fixed upstream model config
must have all non-adapter dropout probabilities equal to zero.

After the common trainable initialization has been hashed, the supported
contract registers exactly three repository-owned stochastic mechanisms: the
CPU `canonical_epoch_permutation`, the replacement adapter-dropout module, and
the non-module `alignment_noise_bank`; all pass explicit seeded generators. A focused AST
inventory scans all Python files under `src/stage1/` for direct calls to
`torch.rand`, `rand_like`, `randn`, `randn_like`, `randint`, `randperm`, `normal`,
`bernoulli`, `multinomial`, `poisson`, `torch.nn.functional.dropout`, Python
`random.*`, and NumPy `random.*`. Only the three registered source locations are
allowed. The same test checks the loaded custom Qwen source for functional
dropout/random calls on the Stage 1 branch; upstream model configuration and
the startup module inventory must disable every remaining dropout path.

Dataset preprocessing, target selection, and teacher inference use no random
API in this audited source scope. This is a static/runtime-configuration audit
of the fixed source digest, not a promise to intercept arbitrary calls inside
PyTorch, CUDA, or third-party kernels. Their numerical nondeterminism remains
inside the explicitly permitted GPU boundary and is measured by same-arm A/A.

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
non-overwritten latent-token embedding gradients remain live. Positions
overwritten by frozen visual features or detached EMA targets intentionally
have no path to their corresponding input-embedding table rows; decoder-output
gradients at those sequence positions remain live. Frozen visual execution
remains under `no_grad`. Row extraction preserves the exact unpadded sample
inputs used by B=1; B=2-only right-padding is introduced only while packing the
decoder inputs.

The implementation first creates `token_embeds = embed_tokens(input_ids)` with
autograd enabled and initializes detached
`replacement_embeds = torch.zeros_like(token_embeds)`, then writes the BF16
samplewise visual outputs and clean EMA targets at their destination positions.
With `replacement_mask` true exactly at those visual/selected-latent positions,
it forms `inputs_embeds = torch.where(replacement_mask.unsqueeze(-1),
replacement_embeds, token_embeds)`. This gives zero input-embedding-table
gradient at overwritten positions and preserves it elsewhere without an
in-place autograd mutation. It calls the composed PEFT student
once with the original `input_ids` for position construction/bookkeeping, the
prebuilt packed `inputs_embeds`, attention mask, explicit multimodal position
IDs, and both `pixel_values=None` and `latent_hidden_states=None`. The Qwen
wrapper path is required to skip token embedding, visual execution, and any
internal image/latent scatter whenever `inputs_embeds` is supplied. The
processor, visual module, and teacher use the same BF16 autocast policy in both
arms.

A causal regression spy verifies that the student visual encoder is called
exactly once per batch row before the composed student call, zero times inside
that call, the wrapper receives an `inputs_embeds` tensor identical to the
packed tensor, and exactly one trainable decoder invocation occurs.

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

The `training_step` stochastic/evidence context owns all samplewise visual and
teacher intermediates until backward completes, then drops its references in a
`finally` block. There is no cross-step feature cache or per-latent-block
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
- with `tau=0.999`, step 1 performs one EMA update and step-2 target
  construction reads that post-step-1 EMA state rather than the initial state;
- resume from a real optimizer-boundary checkpoint restores the exact next
  epoch/update cursor, slot-to-rank and microbatch/row assignments, and only
  then recreates the next step's stochastic tensors from those restored IDs.

### 8.2 Tiny end-to-end witness

An equal initial tiny Qwen/DoRA model trains for at least two optimizer steps
under B=1/GA=4 and B=2/GA=2. The witness compares canonical slot/source
identities, CE, alignment, total loss, accumulated trainable gradients,
optimizer state, scheduler state, student parameter deltas, and EMA parameter
deltas.

At both completed steps, all floating CE, alignment, total-loss, accumulated
gradient, optimizer-state, student-delta, and EMA-delta tensors use
`torch.testing.assert_close(rtol=1e-6, atol=1e-7)`. Exact integer, boolean,
sample-identity, selected-index, scheduler-step, optimizer-step, and EMA-step
state uses equality. The deterministic tiny fixture permits no post-step-1
trajectory divergence; it is a framework arithmetic witness, not the real 3B
acceptance threshold.

The two-step fixture also provides an independent EMA-order witness rather
than only comparing the arms. It snapshots initial teacher `E0` and post-step-1
student `S1`, computes expected state in the production dtype and operation
order as `E1 = E0.clone(); E1.mul_(0.999).add_(S1, alpha=0.001)`, and requires
the live EMA parameters before step-2 target construction to equal `E1` and
differ from `E0` on at least one sentinel element. A teacher-forward spy then compares the actual step-2 target
to a direct target call using `E1`; its EMA-update counter must be exactly one.
This catches a shared stale-teacher bug that paired B=1/B=2 equality would not.

### 8.3 Real 3B, four-GPU online control audit

The approved multi-node route starts the fresh B=1 and B=2 jobs sequentially
in the `llm_kd` container on `10.1.102.41` and the local node, using four GPUs
total: remote rank 0, local rank 1, two visible GPUs per node,
`MASTER_ADDR=10.1.102.41`, and both NCCL/GLOO interfaces `bond1.102`. It reuses
`run_stage1_training.sh` through `python -m src.multinode train-stage1`; no new
distributed launcher is introduced.

Before compute, each job gathers an exact runtime-route object sorted by rank:
logical node role, machine/local/global rank, GPU UUID and model, driver/CUDA
runtime/NCCL versions, `MASTER_ADDR`, world size, and NCCL/GLOO interface. Its
canonical SHA-256 is `route_fingerprint`. Capacity, replay, and main jobs must
all use the same fingerprint; a different rank-to-GPU or software/network
route is a launch error rather than an observed timing difference.

Before either replay, one exact-route B=2 capacity smoke reduces the risk of a
full B=1 run being wasted on a later B=2 OOM. From the fixed three-epoch
schedule, metadata preflight selects for each rank the pair maximizing each of
four axes: padded decoder length, student original-visual elements, maximum
per-row teacher helper-visual elements, and latent/candidate rows. Ties use the
earliest `(epoch, update, microbatch_ordinal)` and duplicate selections are
deduplicated. Rank zero computes the maximum selected count over all ranks,
rounds it up to the next even count, and broadcasts it. Every rank pads to that
same count by cycling its selected pairs in fixed axis order, using its largest
decoder-length pair when it had only one. Thus all ranks execute exactly the
same one or two optimizer updates and collective calls. The selected pairs run
in at most two optimizer updates—two
microbatches per rank per update—with full forward, backward, clip, fused
AdamW, EMA, ZeRO-1, and production gradient checkpointing, so optimizer state
is allocated.
Comparison anchors are disabled because later ordinary steps do not retain
decoder hidden states. The probe must be finite, avoid OOM, and keep peak CUDA
reserved memory on every device at or below 95% of device-reported total
memory. Step time is reported but is not a gate.

This is a smoke-only job, so it may reuse a preprocessing cache after validating
the TRAIN, processor, image, and cache-schema digests; the main runs do not use
that exemption. Its mutated in-memory weights are discarded, it saves no model
or optimizer artifact, and its distinct run ID writes only metrics under
`results/` and logs under `logs/` using schema
`stage1-capacity-preflight-v1`. A failure is a real resource conflict and stops
before the paired experiment.

This bounded axis probe is not a proof that every later shape is OOM-safe;
kernel workspaces and multidimensional interactions may be non-monotonic. A
late B=2 OOM remains an explicit risk and would yield a partial result after
the completed B=1 run. Exhaustively executing all fixed B=2 pairs would remove
the cost advantage of a smoke preflight and is deliberately not performed.

One bounded B=1 replay and then one
bounded B=2 replay first execute optimizer step 1 from the immutable initial
state and the same 648-step scheduler. Their cross-arm comparison is the
common-state A/B control and numerical gate. A failure stops before either
648-step main run.

Each replay remains configured with the 648-step horizon; a diagnostic callback
exits immediately after completing and saving optimizer step 1. It does not
set `max_steps=1` and therefore cannot change warmup or scheduler construction.
Apart from run/output identity and its bounded one-step exit, each replay's
resolved manifest equals its same-arm main.

On that cross-arm pass, the fresh B=1 main starts from the same state. Its first
step must reproduce the B=1 replay; on that same-arm A/A pass, that process
continues uninterrupted through all 648 steps and records the compact control
trace plus requested tensor anchors. The fresh B=2 main starts only after B=1
completes. Its first step must reproduce the B=2 replay; on that same-arm A/A
pass, its B=1-main/B=2-main first-step tensors must also independently pass the
same fixed A/B gates. Only then does the same process continue uninterrupted
through step 648 while comparing compact exact-control records online against
the completed B=1 trace. There is no planned
checkpoint pause, resume, discarded warm-up, or separate full B=1/B=2 short
pair. The two one-step replays are the only duplicated training compute.
The separate high-risk capacity updates are preflight compute and never enter
a trajectory or comparison metric.

The launch-gating control audit compares:

- exact resolved manifests outside the explicit JSON-pointer whitelist;
- exact runtime-route fingerprint across capacity, replay, and main jobs;
- exact canonical global slot/source identities and slot-to-rank placement;
- exact arm-specific microbatch and row grouping against Section 4;
- exact per-slot hashes of unpadded input IDs, labels, attention masks,
  original/helper pixel tensors, and image grids at every step;
- exact per-slot stochastic keys and canonical mask/noise specifications at
  every step: stable site, seed, valid shape, dtype, probability/scale, device
  generator algorithm, and RNG schema;
- exact first-step selected counts and candidate indices;
- samplewise first-step student visual embeddings and initial EMA teacher
  target rows before decoder packing;
- exact first-step decoder position IDs at each slot's valid token positions;
- first-step final decoder hidden states at every canonical valid token before
  `lm_head`, flattened pre-clip trainable gradients, and trainable student,
  EMA, and logical Adam first-/second-moment deltas;
- exact optimizer, scheduler, global-step, and EMA-update counters;
- measured wall time per optimizer step and peak allocated/reserved GPU memory.

The real 3B trace never copies complete dropout masks or noise banks to CPU
solely for hashing. Their tensors are compared directly in the focused CUDA
tests, and their effect is included in the tiny accumulated-gradient witness
and real first-step gradient/update gates. During the 648-step runs, equality
of their complete deterministic specifications is the exact control.

Each main arm's first step and its same-arm A/A replay must match floating
tensors with `rtol=1e-6` and `atol=1e-7`; the gated floating scope is visual
embeddings, teacher targets, final decoder hidden state, CE, alignment, total
loss, accumulated pre-clip gradient, student delta, EMA delta, and logical Adam
first and second moments.
Timing and memory are measured observations, never A/A equality fields. Exact
identities and discrete states use equality. Before the step, this includes
base and trainable initialization, initial EMA snapshot, empty optimizer state,
scheduler state, data/sampler state, and all 16 canonical slots of the first
global update. If either same-arm comparison fails, the corresponding main is
not reproducible under the fixed gate and the B=2 main does not continue past
step 1.

The cross-arm gates below first compare B=1 and B=2 replay step 1 before either
main launch. After both main same-arm gates, B=1-main/B=2-main step 1 must pass
those same fixed cross-arm thresholds before B=2 continues.

Samplewise frozen-visual embeddings and first-step EMA teacher target rows in
B=1 versus B=2 execute through the same unpadded B=1-shaped calls from
identical state and must satisfy `rtol=1e-6` and `atol=1e-6`. First-step
selected counts and candidate indices must agree exactly.
Their cross-arm and both same-arm errors are reported separately. Therefore the
first-update result is explicitly conditioned on accepted upstream same-shape
GPU variation; it is not labeled a decoder-only causal estimate.

For each nonzero first-step tensor category listed below, BF16
numerical acceptance requires both:

```text
symmetric relative L2 error <= 2.0e-2
cosine similarity           >= 0.9998
```

For two FP64 comparison vectors `a` and `b`, symmetric relative L2 is
`2*||a-b||_2/(||a||_2+||b||_2)` and cosine is
`dot(a,b)/(||a||_2*||b||_2)`. A category with both norms zero must match
exactly; a one-sided zero fails. Parameter-shaped canonical vectors concatenate
tensors in lexicographically sorted fully-qualified parameter-name order,
preserving shape metadata. The decoder-hidden vector concatenates each slot's
valid-token rows in global permutation order and then token/hidden-index order;
right padding is excluded. The independently gated categories are: final
decoder hidden state, accumulated pre-clip gradient, student update delta, EMA
update delta, Adam first moment, and Adam second moment. Every category must
pass both thresholds. Per-parameter errors and the worst name are diagnostic
only.

The fixed A/B thresholds are never widened, rescaled, or replaced by an A/A
result. For every A/B category the report places its metric beside the same
metric from both local A/A comparisons. Error ratios are diagnostic only:
relative-L2 and scalar-absolute errors use
`error_AB / max(error_AA_B1, error_AA_B2, 1e-12)`, while cosine uses the deficit
`(1-cos_AB) / max(1-cos_AA_B1, 1-cos_AA_B2, 1e-12)`. No raw cosine ratio is
reported, and none of these ratios is an additional launch gate. The report
may not subtract A/A from A/B or interpret the excess/ratio as a batching-caused
effect or a probabilistic nondeterminism bound.

"Pre-clip" means after GA scaling, BF16 loss unscale, and distributed gradient
reduction but before the configured norm-1.0 clipping. ZeRO-1 logical Adam
moments and gradients are reconstructed on rank zero in canonical parameter
order solely for the checkpoint-tree comparison artifact. Student/EMA deltas
are measured against the immutable common initialization after the optimizer
and one EMA update have completed.

The evidence recorder uses these exact snapshot boundaries. Before the first
forward it captures trainable
initialization, the copied EMA initialization, empty logical Adam state, and
scheduler/counters. During target preparation it captures frozen visual rows
and clean/noisy teacher rows before decoder packing. During only the replay and
main step-1 forwards it captures detached valid-position final decoder hidden
rows before `lm_head`; rank zero orders them at the accumulation boundary.
That boundary also captures the reconstructed gradient after unscale/reduction
and before clipping.
Immediately after fused AdamW `optimizer.step()` and before
EMA update it captures the student delta and logical Adam moments. It then
performs exactly one EMA update and captures the EMA delta, performs the
ordinary scheduler/counter transition, and only then serializes the completed
step trace. Steps 2, 64, and 648 use the same parameter/teacher boundaries but
do not retain full decoder hidden states; ordinary epoch checkpoints are saved
only after the completed-step trace exists.

First-step scalar CE, alignment, and total losses are reconstructed as canonical
global 16-sample means: per-sample values are gathered and reordered by the
epoch permutation offset, summed in FP64, and divided by 16; total loss then
applies the fixed component weights. Each of the three A/B scalar differences
must be at most `1.0e-2`. Exact-zero tensors must agree exactly.
Maximum absolute error and the worst named tensor are always reported but are
not used alone because their scale is not comparable across parameters.
Backward accumulation and reduction order is part of the permitted GPU
batch-shape numerical boundary, not silently treated as exact arithmetic.

The `0.02`, `0.9998`, `0.01`, and same-arm `1e-6/1e-7` limits are predeclared
engineering accept/reject limits, not category-specific bounds established by
prior evidence. The observed `0.0162` decoder-hidden relative-L2 probe only
motivated their scale; it does not prove bounds for gradients, clipping, Adam
moments, student/EMA updates, or separate-process repeatability. The two
bounded replays and main step-1 A/A checks are the first-nail measurements.
Approval of this design accepts that normative choice and the risk that it
stops before a full run. Limits are never widened after seeing a failure. They
do not permit a changed sample, initial target selection, visual input, initial
teacher target, update count, stochastic specification, or loss weighting.

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
canonical dropout/noise specification digests, and
optimizer/scheduler/update counters remain exact controls through step 648.
The B=2 continuation checks these compact controls against the completed B=1
trace step by step and aborts on a mismatch. Gathered alignment-noise rows,
teacher decisions, and numerical trajectory values are observational after
the common first update.

The trace is canonical JSONL schema `stage1-equivalence-trace-v1`, one record
per optimizer step. Each rank submits its four sample records plus rank-local
runtime values. Rank zero requires four unique ranks, gathers exactly 16
unique permutation offsets, sorts samples by integer `permutation_offset`, and
writes one canonical record; duplicate or missing ranks, samples, or steps
fail.

For each completed step, `arm_grouping_digest` is SHA-256 of canonical JSON
over the sorted 16 tuples `(permutation_offset, rank,
rank_local_microbatch_ordinal, row_ordinal_within_microbatch)`. Before writing,
rank zero reconstructs the expected tuple list from Section 4 for that arm and
requires byte equality. The digest is not compared across B=1/B=2 because the
approved grouping is the treatment, but each arm must match its own exact
mapping.

Every record has exactly these top-level fields and JSON types:

```text
schema_version:                 string
manifest_digest:                64-character lowercase hex string
normalized_manifest_digests:    object with cross_arm_v1/same_arm_v1 hex strings
route_fingerprint:              64-character lowercase hex string
arm_grouping_digest:            64-character lowercase hex string
rng_global_step:                integer 0..647
optimizer_step_completed:       integer 1..648
epoch:                          integer 0..2
update_in_epoch:                integer 0..215
counters:                       object defined below
control_digest:                 64-character lowercase hex string
samples:                        array of exactly 16 sample objects
observations:                   object
runtime:                        object
```

`counters` has exactly four integer fields: `optimizer`, `scheduler`, `ema`,
and `trainer_global_step`.

Each sample object has integer `permutation_offset`, `canonical_slot_id`,
`source_index`, `rank`, `rank_local_microbatch_ordinal`, and
`row_ordinal_within_microbatch`; `input_hashes`
with exactly the keys `input_ids`, `labels`, `attention_mask`,
`original_pixel_values`, `helper_pixel_values`, `original_grid_thw`,
`helper_grid_thw`, and `position_ids`; `stochastic_keys` keyed by every sorted
DoRA site plus `alignment_noise_bank`; `dora_mask_specs` keyed by every sorted
DoRA site; string `noise_bank_spec_digest`; hexadecimal-FP64 strings
`sample_ce` and `sample_alignment`; and `teacher_blocks`, an array sorted by
integer `latent_block_index`. Each teacher-block object contains that index,
integer `selected_count`, an ordered integer array
`selected_candidate_indices`, and `selection_margin`, which is either a
hexadecimal-FP64 string or the string `"all_selected"`. The B=1 and B=2 trace
schemas are identical despite their four versus two microbatches per rank.
Every input/spec digest value is a 64-character lowercase hex string, and each
stochastic-key value is the derived unsigned 63-bit seed integer. A mask/noise
spec digest covers the canonical JSON tuple `(schema, site, seed, valid_shape,
dtype, p_or_scale, device_generator_algorithm)`; it is not a hash of copied
full tensor bytes.

`control_digest` hashes canonical JSON containing only `rng_global_step`,
`optimizer_step_completed`, `epoch`, `update_in_epoch`, `counters`,
`route_fingerprint`, and, for
each sorted sample, its permutation/slot/source IDs, input hashes, stochastic
keys, assigned rank, DoRA-mask specifications, and noise-bank specification.
It excludes manifest identity, arm-grouping digest, rank-local microbatch/row
ordinals, teacher blocks, losses, trajectory summaries, timing, and memory.
The excluded grouping fields may differ only according to the closed arm rule
in Section 4 and are validated against `arm_grouping_digest`; other excluded
values are compared separately or observational.

Every finite floating observation is serialized with Python `float.hex()` and
decoded before tolerance comparison; JSON numeric floats are forbidden.
`runtime` contains rank-ordered arrays `wall_time_sec`,
`peak_allocated_bytes`, and `peak_reserved_bytes`. Times are hexadecimal-FP64
strings, memory values are integers, and neither participates in a gate.
`observations` has exactly six hexadecimal-FP64 string fields: `global_ce`,
`global_alignment`, `global_total`, `preclip_gradient_norm`,
`student_parameter_norm`, and `ema_parameter_norm`. Model-bearing anchor
tensors use `torch.save` dictionaries under
`checkpoints/<run-id>/equivalence/`, keyed by the same canonical category and
fully-qualified parameter names. Their derived metrics alone enter `results/`.

The one-step A/A comparisons record one same-arm discrepancy per arm; they do
not estimate a noise floor, upper bound, or repeatability distribution. This
design does not run a third 648-step B=1 replay or multiple paired seeds, so it cannot
separate amplified long-horizon hardware/kernel nondeterminism from a general
microbatch-grouping effect. Final trajectory and accuracy differences are
reported as one controlled paired observation, not as an estimated causal
effect or a generalizable performance delta. That limitation is an explicit
cost waiver, not evidence that long-horizon A/A variance is zero.

## 9. Launch Gate and Artifacts

The bounded B=1 replay may start only after the focused tests and tiny witness
pass; the bounded B=2 replay follows it. A replay/replay cross-arm control or
numerical failure prevents both main launches. The B=1 main stops at step 1
unless its same-arm replay comparison passes and must complete before B=2 main
launch. After the B=2 main first step, a B=2 same-arm A/A failure stops that run
and a direct B=1-main/B=2-main cross-arm failure does the same; neither is
retried at looser thresholds.
On a pass, B=2 continues uninterrupted; later EMA trajectory divergence is
reported but does not fail the run, while any exact controllable-state mismatch
does. Both completed models are then evaluated with one resolved official
evaluation manifest and the same fixed test set. Accuracy and exact
per-example predictions are reported side by side; full-run accuracy equality
is an observed comparison, not a launch threshold, because permitted GPU
numerical differences can compound.

Peak memory and end-to-end/optimizer-step wall time are reported side by side,
including the serialized visual/teacher share where measurable, but there is no
minimum throughput-speedup gate. The controlled comparison may succeed even if
B=2 is operationally slower; that outcome means the semantic isolation worked
but the batching optimization was not useful. Because runs are sequential,
timing remains subject to run-order and system-load confounding and is labeled
an operational observation rather than a controlled speed benchmark.

The final report never collapses the result into one “B=2 success” flag. It
reports three independent outcomes: (1) exact semantic-control agreement,
(2) common-state numerical acceptance under the fixed normative gates, and
(3) operational/quality observations—wall time, memory, trajectory, and
accuracy—with no predeclared utility threshold. “Controlled comparison
completed” means the first two gates permitted both trajectories to finish; it
does not mean “batching optimization succeeded.”

The fresh B=1 arm is the only paired reference. This design deliberately
changes the old unpadded schedule, global RNG consumption, and branched loss
implementation in both fresh arms, so it does not reproduce or quality-gate an
older B=1 checkpoint/result. Prior measured scores may be shown as historical
context but cannot establish continuity with this new controlled protocol.

Official evaluation reuses `run_multinode_evaluation.sh` with four
deterministic source-index shards over the fixed TEST digest and generation
values in Section 2. Prepared inference models live only under
`checkpoints/evaluation-staging/<run-id>/`. The two evaluation manifests may
differ only in model/checkpoint identity and run/output identity. The report
joins predictions by zero-based TEST `source_index`, publishes each model's
exact-correct count over 400 and accuracy, and lists every per-example
agreement or disagreement.

The scorer obtains external labels from each fixed TEST row's
`original_final_answer`, independently normalizes prediction and label, and
does not derive gold answers from either model. However, this same TEST set has
already been inspected in earlier repository experiments and influenced the
decision to rerun training. Its result is therefore a reproducible retrospective
repository-test comparison, not an untouched holdout estimate of
generalization. Establishing the latter would require a newly sequestered test
set and is outside this run.

Artifacts follow repository policy:

```text
checkpoints/<run-id>/   model, optimizer, scheduler, EMA, RNG state
results/<run-id>/       equivalence evidence and comparisons
logs/<run-id>/          execution logs
```

Anchor visual embeddings, teacher targets, decoder hidden states, gradients,
optimizer state, student deltas, EMA deltas, or other model-derived comparison tensors are
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
- a same-arm A/A gate failure;
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
- proving continuity with an older B=1 checkpoint or historical evaluation;
- requiring a minimum B=2 throughput, memory, or quality-per-cost gain;
- proving OOM safety by executing every B=2 pair before the main runs;
- estimating a general batch-grouping effect across full-run A/A replicates or
  multiple seeds;
- Stage 2 batch-size work;
- a new benchmark framework, retry framework, or experiment database;
- starting either main before the replay/replay A/B gate passes, or starting
  B=2 main before the completed B=1 trace and B=1 A/A gate.

## 13. Completion Claim

Completion means the repository can produce fresh evidence that B=1/GA=4 and
true B=2/GA=2 use identical manifests outside the approved batch/output
whitelist and receive the same controllable sample inputs, deterministic
stochastic specifications, objective definitions and weights, schedules, and
update boundaries through the online audit. Actual mask/noise draws are
directly compared in focused CUDA tests rather than copied every real step.
Post-step-1 objective values are trajectory observations and need not agree.
Their common-state first update is accepted only within the fixed upstream,
forward, backward, and optimizer thresholds, with each A/B metric reported
alongside the same objectively defined local A/A metric for both arms. The A/A
values do not change the fixed A/B gates.
Completion also includes observing the resulting EMA trajectory through all
648 steps, identifying any first teacher-selection divergence, and reporting
the paired official evaluations as a single paired observation. It does not
claim identical post-update teachers, bitwise-identical checkpoints, identical
accuracy, or a causal/generalizable endpoint effect without a full-horizon
A/A or multi-seed baseline. Semantic-control, common-state numerical, and
operational/quality outcomes remain separate report sections.

## 14. Design Commitment v20

### Agreed problem and scope

The task is to make fresh Stage 1 DoRA B=1/GA=4 and true B=2/GA=2 runs share
every controllable training semantic at world size four/global batch 16, then
measure their common-state first-update difference and one paired 648-step EMA
trajectory. It is not strict trajectory equivalence, historical-checkpoint
reproduction, or a guaranteed throughput improvement. Stage 2 is outside this
change.

### Final design

Both arms use the fixed contract in Section 2, canonical 3,456-slot schedule,
identical slot-to-rank placement, arm-validated grouping, unified per-sample
losses, sample-keyed DoRA dropout/noise, sequential samplewise frozen vision
and EMA teacher, and one true B=2 trainable decoder call. The run order is:

```text
focused tests and two-step tiny/EMA witness
-> bounded high-risk B=2 capacity smoke (at most two updates)
-> B=1 one-step replay
-> B=2 one-step replay and fixed replay/replay A/B gate
-> uninterrupted B=1 main after its step-1 same-arm gate
-> uninterrupted B=2 main after same-arm and direct main/main A/B gates
-> paired official evaluation
```

Exact semantic controls continue online through step 648. Numerical teachers,
selections, losses, parameters, timing, memory, and accuracy after step 1 are
observations. The report keeps semantic-control, common-state numerical, and
operational/quality outcomes separate.

### Changes from proposal v1

- A real 3B probe disproved batched frozen vision/teacher invariance, so both
  now execute samplewise while only the trainable decoder remains batched.
- The 3,450-row schedule is padded deterministically to 3,456, giving exactly
  216 updates/epoch and 648 total for both arms.
- CE/alignment are per-sample reductions; grouping-dependent dropout/noise are
  replaced by canonical sample-keyed operations.
- Autograd-safe `torch.where` composition replaces ambiguous in-place embedding
  mutation, and checkpoint replay receives an explicit stochastic phase.
- Rank placement, exact B1/B2 pair/row order, route identity, manifests, trace
  schemas, EMA update ordering, and snapshot boundaries are executable gates.
- Both one-step replays and the capacity smoke precede either expensive main;
  both actual main first steps are also gated.
- “Equivalence” was narrowed to a controlled paired observation. Historical
  continuity, utility, causality, and generalization are not implied.

### First nail and status

The v1 first nail was run on the actual 3B model and **failed** the batched
vision/teacher premise: worst teacher-target relative L2/cosine were
`0.598/0.817`, language-hidden relative L2/cosine were `0.744/0.711`, and
candidate selections differed. Supplying exact samplewise visual embeddings
reduced decoder hidden difference to relative L2 `0.0162`, cosine `0.99987`.

The current load-bearing first nail is the bounded B=1/B=2 replay pair followed
by each actual main's step-1 same-arm and main/main gates. It has **not run**
because the sample-stable/grouping/evidence implementation does not yet exist.
The fixed `0.02`, `0.9998`, `0.01`, and same-arm `1e-6/1e-7` limits are
normative engineering choices, not proven gradient/update bounds. They will
not be loosened after observation; failure stops before further expensive work.

### Material divergences and resolved disagreements

- Fresh reviews agreed the result is a controlled characterization, not an
  equivalence proof or batching deployment decision; naming and completion
  claims now reflect that.
- Full-horizon A/A or multiple seeds would be required for causal endpoint
  attribution. They are omitted, so post-step-1 differences remain one paired
  observation.
- One A/A observation per arm is not a noise floor. Ratios are descriptive and
  may not be subtracted or interpreted as batching-caused effects.
- The actual main/main first step now passes the same fixed A/B gates; replay
  transitivity is not assumed.
- Slot-to-rank placement and arm-specific microbatch/row order are independently
  validated; the grouping whitelist cannot hide a wrong pairing.
- EMA correctness is tested against independently constructed `E1`, preventing
  both arms from agreeing on the same stale teacher bug.
- The capacity smoke samples four risk axes and equalizes collective counts
  across ranks. It reduces but cannot eliminate late-OOM risk.
- The evaluation has independent TEST labels, but prior inspection makes it a
  retrospective repository-test comparison rather than untouched validation.

### Unvalidated premises and explicit risk waivers

- The actual PEFT target topology must present `[B,T,H]` activations and accept
  the non-reentrant checkpoint `context_fn`; implementation tests may disprove
  this premise.
- First-step gradients, Adam moments, and updates may exceed the normative 2%
  gate even though the earlier decoder-hidden probe was `0.0162`.
- Separate-process same-arm tensors may fail `1e-6/1e-7`.
- The at-most-two-update capacity smoke is not exhaustive; a later B=2 OOM may
  leave only the completed B=1 result.
- No full-horizon A/A, multiple seeds, historical B=1 reproduction, minimum
  speed/memory/accuracy utility threshold, or untouched holdout is authorized.
- Sequential timing is load/order-confounded, and serialized vision/teacher
  may remove most or all B=2 speed benefit.

Explicit approval of design commitment v20 accepts these limits and authorizes
the implementation plan and staged experiments in this exact order. It does
not authorize threshold relaxation, extra full-run replicates, or broader
infrastructure without a new design decision.
