# Stage1-v5 GPU utilization: design proposal v1

Status: self-reviewed proposal; final human approval pending. Freeze this file
before critique; synthesize any changes in a separate commitment. No product
code or training is authorized by this document alone.

## 1. Problem, scope and authority

Reduce host orchestration and small-operation overhead in Stage1-v5's replay
mask, source-coordinate dropout, student attention observation and Text-off
selector. The user chose these four paths first, leaving rank redistribution
for later. Preserve the existing learning objective, Teacher and reference,
sampler order, rank assignment, accumulation windows and sample progress.
Scope and numerical policy were explicitly answered; architecture A1 was
confirmed by `진행해`, and error/testing section B1 by `계속`. The prior user
selection of Subagent-driven implementation remains in force.

Source baseline: `65643f08cb6e61fbbb200de825d57086f5c7a127`.
Isolated, host-managed workspace: `/root/.codex/worktrees/d3c8/ILVR`.
Existing canonical, 6b9d and 1061 workspaces and historical artifacts remain
untouched. In particular, the completed content-metric run's checkpoint-648
is preserved. No new training, evaluation, dependency installation, sampler,
CLI query mode, loss coefficient or execution framework is part of this work.

The alternatives already presented were four operator paths first (chosen),
rank redistribution in the same change (deferred), and mask/dropout only
(insufficient for the chosen first scope). The chosen approach targets four
bounded internals in the existing execution flow rather than introducing a
compiler, graph cache or launch infrastructure.

## 2. Observed evidence and its limits

The previous run's 36-second inspection averaged roughly 58--64% GPU busy time,
with main-process CPU above one core and data workers near 1%. Process CPU
includes runtime/driver/communication threads. No stage/CUDA profiler attributes
the idle intervals to a specific function. These observations motivate
investigation; they do not establish that the four operators dominate update
time or that optimizing them improves distributed throughput.

An unchanged focused baseline ran with the repository venv and CPU:
64 passed, 2 CUDA-only skips in 12.97s; command and full-output digest are in
the workflow record. This is baseline evidence, not candidate validation.

## 3. Runtime architecture and interfaces

Keep the actual `CustomTrainerStage1V5.compute_loss` flow: build targets;
clean F with per-layer trace and attached K/V; immutable reference logits;
Text key selection; independent packed V-off/Text-off replay; current loss
reductions; one ordinary parent backward. Observation stays detached, while
clean-prefix K/V and the existing decoder/DoRA computation retain gradients.
Keep current public return shapes and source-coordinate meaning.

### 3.1 Replay mask: tensor construction

In `stage1_v5_replay.py`, replace per-suffix-query Python assignments with
broadcast prefix visibility, suffix causal visibility and indexed exclusions.
Retain packed row order: V-off rows first, then Text-off rows, preserving active
block order. Prefix lengths, padded suffix validity, selected logits at
`target - 1 - block.end`, and original multimodal position IDs stay unchanged.

Prefix keys must satisfy source padding and the row's original prefix boundary.
Suffix keys remain causal. Invalid padded queries retain only the original
key-zero escape to prevent all-masked softmax; they never affect valid queries.
V-off excludes only latent body `start+1:end`, leaving START/END and the
original image accessible. Text-off excludes only the selected original keys
for that predictor. Preserve exact 0/-infinity entries and bias dtype.

The existing dense output bias remains; avoid a second dense floating bias.
Use broadcast-sized boolean/coordinate intermediates and indexed in-place
updates instead of constructing a full additional row/query/key workspace.
Keep existing prefix concatenation and its gradient attachment.

### 3.2 Source-coordinate dropout: indexed assembly

In `v5_execution.py`, replace token-by-token Python tensor selection/stacking
with tensor gather/index assembly. A short per-referenced-source loop may
remain. Compact source identity reads may remain where necessary; do not claim
all host synchronization has disappeared merely because the token loop is gone.

For every referenced source and stochastic site, preserve the private
`torch.Generator`, seed from base seed/optimizer step/source slot/site, and the
full original `[source_length, hidden_size]` FP32 random draw before thresholding.
Do not draw only requested tokens or change global RNG state. Gather by the
original token coordinate, preserving duplicates, permutations and empty input.
Keep current native division and packed multiplication scaling arithmetic,
valid/padded treatment, graph ownership, checkpoint recomputation and cleanup.
Do not retain a persistent cache of masks for all layers/sites.

### 3.3 Student attention observation: bounded query chunks

In `stage1_v5_trace.py`, group queries by sample and compute them in bounded
chunks within each layer. Retain native multimodal RoPE, grouped-KV expansion,
source padding, original query/candidate positions, FP32 observation with
autocast disabled, full causal softmax denominator, head mean and sequential
layer accumulation. Gather candidates only after the full-key normalization.
Do not truncate keys or change layers/heads to save work.

The initial internal chunk policy caps at 32 queries and targets 32 MiB for
the two FP32 score/probability tensors, using `2 * heads * chunk * keys * 4`
bytes and a minimum of one query. Thus the unavoidable one-query workspace is
the lower bound when it alone exceeds that target. Masks and compact output
rows are accounted separately in measured peak memory. No new user-facing
flag is needed. Tests must cover a chunk boundary and a short final chunk.

Batch GEMM can change floating values. Numerical tolerance on observed rows
does not permit changing selected original tokens. Any discrete mismatch in
joint comparison is a failed candidate, not a reason to loosen the contract.
No universal exact-selection guarantee follows from finite test coverage.

### 3.4 Text selector: GPU ordering, original boundary arithmetic

In `v5_selection.py`, use stable tensor sorting over detached candidate weights
and retain only the prefix needed by the existing cap. Valid V5Block candidate
coordinates are already sorted and unique, so stable equal-weight ordering
implements the original-position tie-break. Unsorted or duplicated metadata
continues to fail validation; no new metadata contract is introduced.

Preserve per-query original-device FP32 `weights.sum()`, existing full-vocabulary
temperature-squared KL and per-sample rho. Transfer ranked position identities,
needed weights, original masses and rhos in a bounded bulk handoff. Keep
positions as exact integer identities rather than packing them into FP32.
Preserve direct callers with CPU/CUDA attention rows independent of logits.

Keep Python-double sequential `total += weight` and `rho * mass` for the
shortest-prefix decision. Do not substitute a parallel FP32/FP64 cumulative
sum. Retain the low-mass branch, floor-based fraction/cap, zero-cap and empty
behavior. Public results remain original-position tuples in ranked order.

## 4. Execution identity, resume and errors

Current runtime source identity binds eight Teacher/query/Trainer modules but
omits the four modules changed here. Extend current identity and the existing
`verify_stage1_v5.py` source-path bindings to include all four actual modules,
with an explicit execution operator-profile identifier and chunk policy.
Keep the objective schema, formula, coefficients and query semantics unchanged.
Update the actual admission/resume consumers and their literal expectations;
do not merely update declared constants or a helper-level unit test.

A training resume with an old or changed execution identity must fail before
optimizer/training work, with guidance to use its original reviewed revision.
Do not normalize away hashes, reinterpret saved settings, rewrite historical
sidecars, or silently migrate old checkpoints. Historical contracts remain
loadable for read-only inspection under their original semantics; strict new
runtime admission is separate from historical parsing.

Invalid coordinates/shapes, nonfinite values and lifecycle/ownership failures
retain current checks and exception cleanup. Review both debug and ordinary
paths. An equivalence failure is corrected and independently reviewed through
the selected method. If batching cannot meet exact discrete comparisons, stop
that activation and return a concrete design revision; do not silently claim
four-path completion after shipping only three paths.

## 5. Independent oracle and numerical acceptance

Keep the existing frozen historical throughput sources and literal native
full-sequence replay oracle. Add an immutable copy of the four baseline modules
from source65643f0 with recorded hashes. Do not construct expected outputs
using the new candidate's selection/mask/gather logic. Baseline hashes:

| Module | SHA256 |
| --- | --- |
| `src/stage1/v5_execution.py` | `0abd7e9fb51f327916b735b74d7cad53dd0d4b7242d3dc74725dc43257c19a69` |
| `src/stage1/v5_selection.py` | `1b285303a6da390c4de2635aff46c57582272f952d75efc72b52a67eddce2850` |
| `transformers/.../stage1_v5_trace.py` | `33f55a0f6480bbfc50b9256cceb71c6ab61a968cbf4b6815c5dee8f80f5f7efb` |
| `transformers/.../stage1_v5_replay.py` | `bf15d87ea7d6f9a2dd7775c6e1ab4d4b4edb3750bc4096213b8b3e310135fa5e` |

Predeclared before candidate results:

| Comparison | Acceptance |
| --- | --- |
| Selected original keys and order | Exact |
| Replay mask including padding and escape key | Exact |
| Raw dropout masks and unchanged global RNG state | Exact |
| Each detached FP32 attention candidate row | ATOL `1e-7`, RTOL `1e-5` |
| Native FP32 logits, each loss and parameter/prefix gradients | Existing ATOL `1e-5`, RTOL `1e-4` |

Do not weaken tolerances after a failing run. Report dtype/backend, maximum
observed errors and exact discrete results. Tiny FP32 or BF16-projection
operator fixtures do not prove production BF16 training equivalence.

Required cases include ordinary and degenerate inputs, multi-sample/block
interleaving, GQA/nontrivial RoPE, padding, empty and zero-cap selection, exact
ties, `nextafter` neighbors at cumulative and low-mass boundaries, source-
coordinate duplicates/permutations, variable source lengths, chunk remainders,
and checkpoint disabled/full/early-stop paths. Malformed metadata remains
rejected. Direct-selector comparisons use identical attention inputs; joint
trace/selection comparisons begin at identical projections and logits so a
selector-only pass cannot hide attention-induced selection changes.

Exercise real native tiny-Qwen replay and actual Trainer.training_step with
nonzero DoRA paths, each loss and gradients, reference immutability, one F/one
packed/one ordinary backward, and incompatible-resume rejection before training.
Retain existing tests not affected by the approved numerical-policy change.

## 6. Performance evidence and completion

Reuse/extend the current attention-selection operator benchmark for the four
paths; no new benchmark framework. Compare source65643f0 with candidate on
identical inputs. Record warmup3/repetitions20, synchronized raw wall times
and median, host transfers/operator counts, and peak allocated/reserved memory.
Include original fixed geometry and bounded long/multi-block geometry informed
by the existing data audit. No new training model or dataset inference is needed
for operator measurement. Run CUDA checks only after final implementation
approval and a current idle-device/resource check.

Report attention, selector, mask and dropout separately so a regression cannot
be hidden in an aggregate. Improvement requires measured wall-time evidence;
removing Python loops or increasing GPU busy percentage is insufficient.
No numeric speedup is promised before measurement. If no meaningful gain is
observed, report that result rather than claiming the utilization issue solved.

Use TDD and the retained Subagent-driven implementation method, independent
specification/quality reviews, applicable final review, then fresh verification.
The applicable finish gate's project full-suite command runs once on the
reviewed revision, alongside only distinct claim-specific operator/integration
checks. Store test model artifacts under checkpoints, measurements under
results and all logs under logs, using this worktree's existing links/venv.

CPU correctness, CUDA operator performance, production BF16 equivalence and
actual two-node/four-rank training speed are separate evidence levels. This
implementation scope does not itself launch a new training comparison.
A later authorized training measure uses the existing mandatory coordinator,
four-rank runtime participation witnesses, unchanged sampler/settings and
matched seconds/update or tokens/s. No single-node training fallback.

## 7. Remaining uncertainty before final commitment

No candidate, profiler or premise experiment has run. It remains uncertain
whether these paths materially limit distributed throughput, whether chunked
attention preserves selected keys on production boundary cases, and what
actual extra memory and speed tradeoff results. Source inspection and baseline
tests cannot resolve those claims. The required hate/macrothink critique is
next; its proposed experiment is not authorization to run it. The exact final
commitment must state experiment status and obtain explicit human acceptance
of any untested load-bearing uncertainty before implementation.

## 8. Inline self-review record

Context and actual runtime interfaces checked; all four paths, ownership,
source identity consumers and acceptance witnesses are specified. Corrected
the earlier unsorted-metadata test wording to unchanged rejection after
reading V5Block validation. No placeholders, new unsupported input contracts,
formula changes or implicit training permission remain. Numerical thresholds
were fixed before any candidate GREEN result. This is design self-review,
not independent implementation review or a runtime equivalence claim.
