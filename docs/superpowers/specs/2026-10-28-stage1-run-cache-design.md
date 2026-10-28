# Stage 1 experiment-scoped computation cache — proposal v1

Status: written brainstorming proposal; **not finally approved and not implemented**.
Source baseline: `5faca3b63b26ba7e6ab7e33d148cd9e5bc0b2335`, worktree
`/root/.codex/worktrees/f187/ILVR`. This is a Superpowers control artifact.
Conversation decisions and supporting evidence are recorded in
`logs/stage1-cache-design-20261028/intake.json`.

## Intent and constraints

Reduce repeated Stage 1 preprocessing, frozen helper computation and frozen
reference decoding across epochs, preserving the existing training result.
The user requires bitwise-identical cached values, selection indices, losses,
gradients and optimizer updates, with unchanged sample order and RNG behavior.
This is an acceptance requirement, not a demonstrated property of the design.

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
Their latest request asks whether any important decisions remain. Written-spec
approval, any GPU premise-probe authorization, and implementation authorization
remain outstanding.

## Resolve cache eligibility once at launch

| Path | Persisted payload | Work that remains online |
|---|---|---|
| Common preprocessing | Static token IDs, labels, masks, original image pixels/grid and static sample descriptors | Dynamic padding, occurrence/rank/epoch ownership and batch offsets |
| Fixed helper teacher | Final chosen targets, counts, selected indices, selection margins and any enabled reverse donor values/provenance | Student forward and losses |
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

## Evidence and acceptance

Bitwise equivalence is currently **unverified**. Before expensive implementation
commitment, obtain an authorized bounded premise probe: check that the unchanged
frozen computation itself repeats bitwise on the same inputs and across both
nodes using the actual backend/geometry. A baseline A/A disagreement defeats the
assumption that a first-epoch value can represent every later recomputation.
Do not silently relax to an epsilon or change the execution backend to pass.
The GPU probe is not authorized by this document. Its exact bounded procedure
must be presented separately before execution; keep the required coordinator
and two-node/four-rank topology for any training probe.

Implementation validation must then compare uncached/cached A/B with identical
model, optimizer, RNG and batch state: payload bits and selection indices, losses,
gradients, optimizer/scheduler updates, EMA boundaries, reference isolation and
runtime metadata. Cover first-write, read-only replay, node exchange, epoch
transition and resume/missing-cache preparation. Include actual four-rank
evidence; small CPU contract tests alone do not establish runtime equivalence.
Independent review must trace actual calls and hooks, not infer them from names.

Measure steady-state updates/sec and data/transfer waits, as well as first-epoch
write overhead, node exchange, actual disk/RAM use and end-to-end amortization.
Skipping GPU work is not itself a speedup result. Report observed vs projected
whole-run times separately. Do not launch a new 10-epoch experiment merely to
validate the cache without user authorization. Use one focused implementation
test set, the smallest necessary integration witness, and any workflow-required
final verification once rather than repeated broad cycles.

## Remaining commitment gates

Builder check is pending. After it, freeze the complete proposal and perform the
required `hate` and applicable bounded `macrothink` passes on that same version.
Synthesize their evidence and unresolved disagreements into a design commitment.
No success claim for numerical equivalence or speed, final written-design
approval, implementation plan, product code, GPU probe, training launch, storage
migration or deletion is implied by this control document.
