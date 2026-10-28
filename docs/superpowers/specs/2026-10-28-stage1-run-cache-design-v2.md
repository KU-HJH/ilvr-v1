# Stage 1 experiment-scoped computation cache — proposal v2

Status: numerical policy and production-check policy approved in conversation;
complete written commitment pending final review. Cache implementation has not started.
Source baseline: `c5d6280ec661d8e4fe48f5e1f01f1ec4e06300b3`, worktree
`/root/.codex/worktrees/f187/ILVR`. This is a Superpowers control artifact.
Conversation decisions and supporting evidence are recorded in
`logs/stage1-cache-design-20261028/intake.json`.

## Intent and constraints

Reduce repeated Stage 1 preprocessing, frozen helper computation and frozen
reference decoding across epochs, preserving the existing training result.
The user approved deterministic cache ON/OFF validation with the existing fast
backward retained for actual training. Cached payload bits and semantic selection
results, sample order, RNG behavior, objectives and optimizer/EMA semantics must
remain unchanged. Whole-production-gradient bitwise reproducibility is no longer
promised: the uncached baseline itself showed backward variability. This is not
permission for lossy cache data or arbitrary numerical drift introduced by caching.

The latest instruction also excludes every nonessential check from actual
training. Verification and diagnostic tensor scans belong in the separate
validation path. Necessary data/cache lifecycle and execution safety checks are
placed at the earliest correct boundary and do not repeat per tensor or step.

Support the existing standard Stage 1 fixed/EMA teacher paths and frozen-teacher
V5. Do not add moving-EMA support to V5: it currently requires decay 1. Keep the
existing objective, attention backend, helper query/donor semantics, precision,
projection geometry, sampler, tail denominator and schedule. Fresh runs retain
the standing two-node/four-rank topology and four DataLoader workers per rank.
Saved resume semantics remain authoritative. The implementation method is the
previously selected Native method, with independent review.

No caching of dynamic student logits, entropy-selected positions, vocabulary
top-k, student attention, trainable hidden states, gradients or optimizer work.
No cross-experiment cache reuse, persistent EMA target versions, lossy
compression, dtype conversion for storage, whole-vocabulary logit cache, or
unbounded pinned-memory cache. No active/historical run or checkpoint changes.

## Approaches and decisions

The selected computation approach uses the original first-epoch calculation and
lossless replay. Preprocessing-only caching leaves helper/reference work intact;
the selected layered approach also avoids their repeated fixed computation.
Full reference-logit caching was set aside for storage/IO cost, not because it
is inherently invalid. The user subsequently selected final reference hidden
state caching with the existing frozen reference `lm_head` still executed.

For storage, full node-shared RAM residency plus disk backing and disk-backed
read-only mapping with OS page cache were discussed. **Builder recommendation:**
use SSD files plus read-only mmap/OS page cache. Do not silently claim this
provides guaranteed residency or a cache-specific RAM cap. There is no separate
whole-cache Python dictionary in each rank or worker. Pin only bounded batches
for GPU transfer; retain the existing sampler/prefetch ordering.

The user accepted first-epoch generation, later read-only use, experiment-only
ownership, resume reuse and deletion on normal training/evaluation completion.
The latest instruction approves validation-only determinism, fast production
backward and removing nonessential checks from training. These choices must not
be requested again. Final review concerns this complete written commitment and
any unresolved empirical premise, not those already settled choices.

## Resolve cache eligibility once at launch

| Path | Persisted payload | Work that remains online |
|---|---|---|
| Common preprocessing | Static token IDs, labels, masks, original image pixels/grid and static sample descriptors | Dynamic padding, occurrence/rank/epoch ownership and batch offsets |
| Fixed helper teacher | Final chosen targets, required counts/geometry and enabled reverse donor values/provenance; optional native metadata only when already produced | Student forward and losses |
| Updating EMA helper teacher | Frozen visual encoder/merger output and its image partition descriptors | Teacher language forward, query construction and chosen/reverse selection on every call |
| Frozen reference with invariant complete inputs | Last decoder hidden states for every reference query position | The reference owner's unchanged chunked `lm_head` and downstream loss/selection |

Do not infer reference eligibility solely from whether another teacher has EMA.
The existing V5 reference receives the detached helper latents in `model_inputs`
and uses the same frozen owner. Standard Stage 1 EMA has no separate V5 reference
forward. Any existing reference path whose weights or complete injected inputs
change is ineligible; determine this from the supported graph/config before
launch. Do not invent a different reference latent source to make it cacheable.

Use all V5 query positions, not only the changing clean-entropy top-20% mask.
Capture reference hidden states immediately before the original `lm_head` and
retain its owner, dtype/autocast, position order, projection chunks and shape.
The native full-vocabulary DIL term and Text-off selector remain unchanged.

Keep helper preprocessed pixel arrays only as transient first-epoch inputs.
After computing the selected persisted payload, do not retain redundant helper
pixels or (for fixed targets) intermediate helper vision features on disk.
Cached target replay rebuilds batch-local masks and donor block batch indices.
Do not enable production diagnostics merely to populate a cache: current debug-off
paths may omit selected-index/margin diagnostic records. Preserve that absence;
extra selection evidence is collected only in validation. Values/geometry that
the actual algorithm consumes remain required, regardless of their field names.
The EMA replay point is the visual output, before the existing candidate/grouping
and query-dependent selection code. Cache outputs are detached and immutable.

## Creation, reading and lifetime

1. Before a fresh run, resolve the fixed policy and a new experiment namespace
   exposed under `data/cache/<run-id>/` on both nodes. Preserve the existing
   `data` symlink. Local `data` currently points to HDD RAID; SSD placement needs
   a dedicated cache subpath pointing at SSD storage, not relocation of datasets.
2. First epoch executes the original computations and saves each enabled static
   payload. Do not add a full-dataset model prepass. Duplicate sampler slots
   retain their original computation; publish a complete record once rather
   than altering the training traversal to avoid duplicates. A bounded atomic
   publication mechanism is sufficient; no general cache service is needed.
3. At the first-epoch boundary, finish writes, verify retained-sample coverage
   and exchange the two nodes' records once. Both nodes must have the complete
   cache before the next epoch, since shuffling can move samples between nodes.
   This is an explicit added IO cost; shared storage is not assumed.
4. From epoch two, readers use stable sample indices in read-only mappings.
   They load only the batch-required tensors and rebuild dynamic metadata. No
   per-step weight hashing, input/output equality comparisons, target-version
   invalidation, or lazy creation in later epochs. OS page eviction merely causes
   a disk read; it does not trigger model recomputation.
5. An incomplete-first-epoch resume continues population. A later-epoch resume
   checks coverage before entering training. Missing derived payloads can be
   rebuilt in a separate preparation phase using the saved initial fixed
   teacher/reference or unchanged vision state. Never recreate a frozen
   reference from the resumed student. Cache loss does not invalidate model
   checkpoints. Never silently reinterpret old resume contracts.
6. On normal completion of training and its scheduled evaluation, close readers
   and mappings, then remove only that run's cache on both nodes. Interrupted
   resumable runs retain it. Preserve results/logs and all model checkpoints.
   No cross-run sweep or disk-pressure eviction of other runs is authorized.

A later-epoch missing/malformed record is an execution error; do not silently
switch the training path or consume partial data. Ordinary OS page misses are
not missing records. Resume preparation is the defined way to rebuild records.
Existing single-epoch runs should not enable this optional optimization by
default: they cannot benefit from the later-epoch replay phase.

## Data identity, execution integration and resource bounds

Reuse existing run/processor/reference identity records. Bind cached static
payloads to the source data and image content, resolved processor, query/donor
configuration, effective frozen weights, native dtype/backend, and schema.
Validate these once before use/resume, not through per-step tensor comparisons.
Write/coverage checks establish each record's dtype, shape, lengths and integrity;
read-only training does not rehash or rescan its tensor contents.
Do not bind static sample records to epoch/rank/padded batch offsets. Readers
must not mutate mapped backing storage when assembling labels or masks.

The implementer must trace the actual Trainer/Accelerate/DataLoader/DeepSpeed
dispatch. Epoch transition must reach the workers before the next iterator
prefetches records; a parent-only flag is insufficient. Preserve all-rank
collectives and forward/buffer side effects required by the existing wrappers.
Do not assume frozen/eval implies absence of mutable RoPE or other forward state.
Support only configurations for which these requirements can be demonstrated.

Choose a simple existing lossless tensor format with CPU mmap support, indexed
by stable sample ID. Format and file grouping are builder discretion; avoid new
dependencies and keep writes and pinned staging bounded. Node-local ranks and
workers share file-backed OS pages; current-batch materialization is allowed.
Do not reserve all 503 GiB of host RAM or promise model memory is negligible.

All 3,450 TRAIN rows and image headers were counted using actual processor resize
geometry and V5 tokenizer/source mapping. Three unmodified CPU collator checks
confirmed the shape/dtype counting method. Estimated payload per node:

| Payload | GiB |
|---|---:|
| Original FP32 pixels | 52.60 |
| Four token/mask/label arrays | 0.115 |
| Fixed chosen targets, up to 8 per block | 0.327 |
| Optional reverse targets in addition | 0.327 |
| Reference hidden, BF16 | 1.621 |
| EMA helper vision output, BF16 | 49.736 |
| Fixed V5 total | 54.7–55.0 |
| Standard EMA total | 102.5 |

If vision output is FP32, the EMA total is about 152.2 GiB. Do not downcast to
meet the estimate. Metadata/file overhead and runtime staging are additional.
Redundant helper pixels would add 228.5 GiB and are deliberately not retained.
Two nodes each retain a complete copy; epochs do not multiply storage. Evidence:
`results/stage1-cache-capacity-20261028/estimate.json`. Observed SSD free space was
556 GiB local and 635 GiB remote; availability is not a reservation or benchmark.

## Essential checks and validation-only work

Resolve the execution/check policy once during construction, then use the bound
production or validation path. Avoid new per-step policy negotiation or fallback
branches. Extend/reuse existing V5 debug-off behavior for the supported standard
Stage1 path, while keeping old/resumed execution identities explicit.

| Boundary | Required work |
|---|---|
| Launch/resume preparation | Existing input/config/checkpoint compatibility; fixed/EMA cache eligibility; supported runtime; worker4/four ranks; path ownership and storage capacity; schema/source identity; complete coverage for replay |
| Record publication / epoch transition | Required shape/dtype/index ranges and complete publication; first-epoch write completion; full retained-sample coverage; bounded node exchange; worker-visible read-mode transition |
| Actual training | Required lookup/ownership and cheap shape/length metadata checks; fail on missing/unreadable records; required sampler/count reductions, backward, clipping and applied-step/EMA ordering; existing framework nonfinite/overflow safeguards |
| Checkpoint/evaluation cleanup | Saved resume identity, coherent checkpoint writes and run-scoped path ownership; close mappings before deletion; retain interrupted runs |
| Separate validation only | Cached/live full tensor equality or hashes; per-step weight/RNG/gradient dumps; whole-intermediate finite scans; recomputation solely for auditing; detailed target/support/margin statistics; timing/profiler hooks; stochastic-site event lists; duplicate mask/target-layout audits |

A diagnostic global norm recomputation solely to assert finiteness must not be
confused with the norm needed by real clipping. Reuse existing framework
safeguards; do not add another pass over all parameters or another all-reduce for
a diagnostic. If a supported path has a correctness-critical failure that its
framework does not catch, retain the smallest necessary check at that boundary
and document the actual reason. Do not create speculative hardening requirements.

CPU/GPU operations that construct the mathematical result are not checks. Keep
alignment's live noise draw, stochastic dropout coordinate/seed construction,
dynamic entropy/top-k selection, required target counts and real collectives.
Removing `.item()`, `.tolist()` or a reduction solely because it synchronizes is
not authorized if it changes the algorithm. Cache immutable metadata to avoid
reconstruction where supported; optimize other math only under separate scope.

No optional diagnostic collection in the production loop, including cache-off
runs using the new execution policy. Full diagnostics remain explicitly
available to the separate verifier. Old run artifacts and saved resume settings
are preserved; do not silently reinterpret their contracts.

## Evidence and acceptance

The authorized uncached V5 A/A probe completed two one-update/four-rank arms.
Observed inputs, helper/reference outputs, losses and RNG matched; actual
optimizer gradients and Adam moments differed. LR was zero on that first warmup
update. This disproves the former whole-production-step bitwise premise, not
cache correctness. Records: `2026-10-28-stage1-cache-aa-probe-results.md` and
`results/stage1-cache-aa-comparison-20261028/`.

Subsequent synthetic SDPA operator probes reproduced unequal dQ with equal
forward/dK/dV without optimizer or communication. Deterministic mode made the
observed repetitions equal. This is operator evidence, not a full-model or
cross-node proof. See `results/stage1-gradient-nondeterminism-20261028/analysis.json`.

Validation runs form their own explicit, isolated policy:

1. Confirm repeated uncached A/A for the exact deterministic validation geometry
   before interpreting cache ON/OFF differences. Record effective deterministic
   flags/backend and initial state. Do not change dtype, mask, chunking or
   attention algorithm just to force a pass; if the intended deterministic
   path cannot repeat, report the specific blocker instead of masking it with
   an invented tolerance.
2. Compare cached/live payload bits and semantic selections, losses, gradients,
   optimizer/scheduler and EMA behavior from identical state under that same
   deterministic policy. Include at least one nonzero-LR update using unchanged
   schedule semantics; LR0 equality is insufficient evidence for updates.
3. Separately confirm that enabling validation-only determinism is not hiding a
   cache producer difference in the actual fast path: compare cached/live frozen
   payloads there with identical inputs/geometry, and inspect whether skipped
   forward calls have RNG, RoPE/buffer or wrapper side effects. Verify output of
   the unchanged chunked reference `lm_head`, not hidden serialization alone.
4. Cover first-write/read-only replay, first epoch boundary and worker visibility,
   cross-node use with shuffled ownership, updating EMA and later-epoch resume
   with missing derived caches. These are bounded acceptance witnesses, not
   per-step production audits. EMA targets themselves must still change online.
5. Verify by runtime counters/observers in the verifier that production mode
   does not call optional comparison, hash, profiling, diagnostic-statistic or
   duplicate finite/norm scan functions, and retains required math/lifecycle.

The full supported validation route and cache A/B results are not yet measured.
Acceptance is contingent on these witnesses; there is no claim of final task
accuracy, convergence or complete full-model determinism. Actual training keeps
its existing fast backend and nondeterministic backward. Numerical tolerance
must not be invented after observing cache differences to declare a pass.

Standard native EMA currently needs the preserved DeepSpeed0.18.0 runtime;
V5 uses the current0.18.7 path. Use existing repository venvs on both nodes, not
package reinstallations. If needed for the required EMA witness, allow only the
explicit supported native-standard profile through coordinator preflight while
retaining exact both-node version/topology/cleanliness checks; ordinary V5 and
Stage2 keep their current profile. Do not enable moving-EMA V5 or change native
standard attention to imitate V5. The separate profiles are each compared
against their own identical uncached baseline.

Measure steady-state updates/sec and data/transfer waits, first-epoch write and
node-exchange overhead, actual disk/RAM use and end-to-end amortization in the
fast production path. Deterministic verifier throughput is not the production
speedup. Report observed vs projected whole-run times separately. No new
10-epoch experiment is authorized merely for cache validation. Use one focused
implementation test set, the smallest sufficient integration witnesses, and
required independent review/fresh verification without repeated broad cycles.

## Remaining commitment gates

The v1 builder check and critiques remain historical evidence, not approval of
this revision. Freeze v2, perform the applicable design critiques and builder
check, then synthesize a complete commitment with evidence and remaining
uncertainty. Final written-design review precedes writing-plans and Native
implementation under the existing policy. This document authorizes no training
launch, storage migration or cache deletion by itself.
