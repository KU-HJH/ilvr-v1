# Stage1-v5 GPU utilization: design commitment v1

Status: self-reviewed final commitment; explicit human approval and uncertainty
acceptance pending. This is the exact implementation approval artifact. It
synthesizes the frozen proposal and the root hate/three macrothink reads.
No product code or training is authorized by this document alone.

## 검토할 최종 결정

- 기존 흐름 안에서 mask tensor 구성, 동일 난수의 dropout gather, 제한된
  query 묶음 attention, GPU 안정 정렬과 기존 Python 누적합을 구현한다.
  샘플 순서·rank 배분·Teacher·reference·loss 계산과 평균은 유지한다.
- 먼저 기존 연산별 시간을 기록한다. Attention 묶음 계산은 독립 비교를
  먼저 통과해야 통합한다. 선택 token이 달라지면 허용 오차로 통과시키지
  않고 해당 변경을 중단한다. 구현하지 못한 경로는 완료로 보고하지 않는다.
- Token 선택·mask·dropout 난수는 정확 일치, FP32 attention은
  ATOL1e-7/RTOL1e-5, 기존 FP32 loss·gradient는 ATOL1e-5/RTOL1e-4다.
- 실행 소스와 연산 profile을 resume 계약에 포함한다. 기존 checkpoint는
  보존하고, 호환되지 않는 코드를 같은 실행으로 조용히 이어가지 않는다.
- 승인 후 기존 Subagent 방식으로 계획·TDD 구현·독립 검토·검증을 진행한다.
  로컬 CUDA 연산 비교는 포함하지만 새 학습 실행은 이 승인 범위에 없다.

**명시적으로 수용해야 할 불확실성:** 제안한 사전 전제 실험과 실제 분산
실행 profile은 수행하지 않았다. 네 경로가 실제 병목인지, attention 묶음
계산이 실제 경계 사례에서도 token 선택을 보존하는지, VRAM과 학습 속도가
얼마나 달라지는지는 아직 모른다. 이 불확실성을 수용하고 제한된 구현과
비교 검증을 진행하는 데 대한 승인이다. 미래의 모든 입력에 대한 동등성,
생산 BF16 또는 2-node/4-rank 학습 속도 향상을 승인만으로 확정하지 않는다.

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

Order after final approval: first preserve the baseline and record its local
operator costs; implement the exact mask/dropout changes with TDD; implement
stable ordering while retaining sequential boundary arithmetic; run a minimal
batched-attention joint differential witness before broad integration of that
path. Do not spend a full training run to discover a local boundary mismatch.
Failure stops the affected path and returns a concrete revision; it does not
authorize silently dropping one path or changing the accepted numerical policy.

Reuse/extend the current attention-selection operator benchmark for the four
paths; no new benchmark framework. Compare source65643f0 with candidate on
identical inputs. Record warmup3/repetitions20, synchronized raw wall times
and median, host transfers/operator counts, and peak allocated/reserved memory.
Include original fixed geometry and bounded long/multi-block geometry informed
by the existing data audit. No new training model or dataset inference is needed
for operator measurement. Run CUDA checks only after final implementation
approval and a current idle-device/resource check.

Include paired baseline/candidate timing order so one implementation does not
always receive a colder allocator or cache. Extend the existing direct script
only as needed. The 32 MiB score/probability target is not a total VRAM bound;
record actual extra peak memory including temporary kernels/masks/outputs.

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

## 7. Critique synthesis and uncertainty acceptance

Frozen proposal v1 is unchanged at commit
`93bd9519a73ab4eda4c1c820ae5a522db508924a`, SHA256
`e036cc923079233ad2ccf4bc0807164c827337742ca303a974aa506a91c9aec0`.
Full root hate result, identical stripped macrothink input, agent identities,
individual results and classifications are in the linked workflow review
`../2026-10-18-stage1-v5-gpu-utilization-design-review.md`.

### Divergent-incompatible findings first

The fresh readers challenge committing to four rewrites before attributing the
critical path. Main-process CPU and busy-time measurements cannot determine
whether orchestration, decoder work, communication or geometry dominates.
Their suggested unchanged-execution CPU/CUDA timeline has not been obtained.
They also note that selection already has a single packed host transfer, so
its benefit cannot be inferred from a generic synchronization argument.

One reader explicitly recommends deferring changed attention GEMM shapes until
joint boundary comparison. Root hate identifies the same load-bearing risk:
allowed float perturbations precede discontinuous selection, so close rows do
not establish exact interventions. A changed selected key falsifies acceptance
of that batching path even when every attention error is within tolerance.

### Root synthesis, compatible contributions and unresolved disagreement

Retain the user's explicit four-path first scope, but measure original local
operators first, stage exact mask/dropout work before attention integration,
and put joint attention/selector differential evidence before acceptance of
the new GEMM shape. Keep original sequential threshold arithmetic and the
existing host-transfer baseline. Report per-operator regressions and memory.
These tighten order/evidence without changing sampler, math or the four-path
interfaces. The alternative priority of profiling a real distributed update
before any rewrite remains a valid unresolved disagreement; no claim is made
that local operator costs substitute for that missing profile.

Fresh reads also support preserving the existing execution shape, RNG draws
and boundary arithmetic. That agreement is reassurance only, not correctness
evidence or proof. Root owns the decision; no vote or average of opinions
authorizes implementation.

### First-nail status and remaining uncertainty

The proposed first nail is a small CUDA operator comparison of frozen
per-query and candidate batched attention followed by the unchanged selector,
starting with identical deterministic projections, padding, candidates and
logits at ties/nextafter threshold boundaries. It needs no model load or
training, but it is implementation-like and was neither separately authorized
nor executed during design. No result is reported for it.

No candidate, stage profiler or new premise experiment has run. Existing
baseline tests do not establish that four local changes will improve useful
training throughput, preserve production boundary selections or fit an actual
full model's memory budget. Those remain unverified. Approval must explicitly
accept proceeding to bounded implementation under this uncertainty; it is
not a waiver of equivalence tests, independent reviews or later launch gates.
Post-approval operator tests will address limited-input correctness/performance
and may fail. Production BF16 and distributed claims need separate witnesses.

Changes from frozen proposal: staged measurement/boundary gating before
integration, paired timing, clarified memory-target boundaries, and the critique/
unresolved disagreement/uncertainty record. No coefficient, objective,
sampler, rank assignment, public input contract or training scope changed.

## 8. Inline self-review record

Context and actual runtime interfaces checked; all four paths, ownership,
source identity consumers and acceptance witnesses are specified. Corrected
the earlier unsorted-metadata test wording to unchanged rejection after
reading V5Block validation. No placeholders, new unsupported input contracts,
formula changes or implicit training permission remain. Numerical thresholds
were fixed before any candidate GREEN result. This is design self-review,
not independent implementation review or a runtime equivalence claim.
The written commitment was rechecked against A1/B1 and the frozen proposal:
additional details concern execution ordering and evidence, not new behavior.
The final human gate, independent implementation reviews and completion
verification remain outstanding; critiques have not been labeled those gates.
