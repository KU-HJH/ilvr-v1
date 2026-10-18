# Stage1-v5 GPU utilization optimization workflow

## Intake and authority

Understood as: improve actual Stage1-v5 throughput by reducing host-side
attention/selection/mask/dropout overhead and investigating rank imbalance,
while preserving the existing query/loss/teacher/reference semantics and
completed training artifacts. The user's `수정진행해` authorizes pursuing this
development; it is not final approval of a subsequently written architecture.

Source revision: `65643f08cb6e61fbbb200de825d57086f5c7a127`. The previous
feature and merge reviews plus full-suite5084pass/11skip remain evidence for
identical source; no broad baseline rerun is needed. Existing data/checkpoint/
venv links are reused. No package install, evaluation or training is requested.

User answered the numerical-policy question: selected token indices, attention
masks and dropout randomness must agree exactly; floating attention/loss/
gradient values may use small, predefined tolerances. Exact tolerances and
edge behavior still need specification. Existing comparisons of scalar logits/
loss/gradients use ATOL1e-5/RTOL1e-4; old throughput attention tests currently
require exact rows. Do not silently relax those tests without the approved
new numerical contract and an unchanged baseline oracle.

The user answered the scope/decomposition question: `연산 경로 4개 우선,
rank 배분은 후속 단계 (권장)`. This confirms attention observation, text
selection, replay masks and source-coordinate dropout as the first scope.
Keep sample order, sampler, rank assignment, accumulation and resume sample
progress unchanged. Rank balancing is deferred to a separate later design;
it is not included in this implementation. This answer confirms scope, not
the subsequently presented architecture or final written commitment.

## Isolation and actual prior-training status

Native worktree list returned no task attachments. Other existing checkouts
contain separate user branches or the prior launch's code/results. Root used
native create_worktree(ref=65643f0), operation
`2e1256dd-c3ba-49e7-b44f-13ae80012e5b`, then observed completed creation of
`/root/.codex/worktrees/d3c8/ILVR`. It is host-managed, detached, same source
revision and initially clean. The user accepted working separately from the
training source in the preceding proposal. Preserve this workspace's host
ownership; do not delete other worktrees, branches or unrelated files.

Root initially treated the previous training as still live. A fresh process/
artifact check found it already complete: both node exits0, producer evidence
global_step648, all process ranks0..3, finite_loss=true, aggregate train loss
1.3436651700808677; trainer_state at checkpoint-648 has epoch3.0. This update
was reported to the user. Existing canonical and6b9d files remain unchanged.
The fresh workspace links venv/data/checkpoints to their canonical existing
locations; no model-bearing data is copied to results or documentation.

## Active lifecycle and task state

Superpowers is controller. Read using-superpowers, brainstorming and
using-git-worktrees, canonical/worktree AGENTS and the prior throughput brief.
Readchk was applied to scope/constraints. Active phase: architectural
brainstorming for a multi-step runtime/numerical optimization. Retain the
previous user-selected Subagent-driven implementation method after approval.
Root authors only control artifacts. No product code, tests or configuration
have been changed; no implementation agent has been dispatched.

1. Context exploration: actual runtime call paths and source/test oracles read.
2. Visual companion: nonapplicable; decisions concern arithmetic, interfaces
   and validation rather than visual layouts.
3. Clarifications: numerical policy and scope/decomposition answered.
4. Alternatives: four-operator scope selected over rank redistribution and
   mask/dropout-only scope. Preserve this choice rather than asking again.
5. Architecture/components/data-flow section: proposal below, awaiting human
   confirmation. Error handling and testing section follows that confirmation.
6. Complete written specification and self-review: pending confirmed sections.
7. Freeze proposal, root hate, applicable macrothink fresh reads, synthesize
   commitment: pending complete unapproved proposal. No premise experiment is
   authorized or reported as run by those advisory hooks.
8. Final human approval of exact written commitment, including any explicit
   uncertainty acceptance: pending. Initial implementation request is not it.
9. Only after approval: writing-plans, retained SDD/TDD implementation and
   independent spec/quality/final review, fresh claim-specific verification,
   then applicable branch finishing. No current performance-completion claim.

## Findings and constraints to resolve in the design

The36-second prior read-only observation averaged58..64percent GPU busy time;
main-process CPU exceeded one core, worker CPU averaged about one percent.
CPU observations include runtime/communication threads and do not isolate
Python time. No stage/CUDA profile establishes a dominant function.

Actual paths: v5_trainer.compute_loss builds targets, then F with per-layer
capture_v5_trace, immutable reference, select_v5_text_keys, packed V/Text-off,
loss and native backward. Trace loops over each query in each layer;
selector bulk-transfers weights and performs Python stable ranking/prefix
accumulation; packed replay fills bias per query; SampleStableStage1Dropout
dispatches through active V5Execution to source-coordinate masks, whose
dropout_mask currently converts coordinates to host and stacks per-token rows.

Potential constraints: preserve full causal softmax denominator and original
candidate coordinates; FP32/head/layer semantics; stable position tie-break,
low-mass/cap/shortest-prefix behavior and Python-double cumulative decisions;
exact 0/-inf masks and source padding; identical random draw shape/seed/site/
step/source identity and checkpoint replay; attached F K/V gradients; immutable
reference and Teacher query rules; all existing loss reductions, GAS, sampler,
workers, resume/source-binding strictness and storage policies. Batched float
math does not itself establish exact discrete selection at a boundary.

No broad compiler, CUDA graph cache, retry/recovery or sampler infrastructure
is assumed. Existing DeepSpeed overlap_comm is already true. Future benchmarks
need identical baseline/candidate inputs, wall time and peak memory as well as
operator count and exact discrete comparisons. GPU utilization alone is not
the success metric. Separate CPU evidence, operator CUDA evidence and actual
two-node/four-rank training throughput; none may stand in for another.

## Fresh focused baseline and additional contract finding

At control-only revision492b0e8, root ran the existing unchanged focused CPU
tests with CUDA_VISIBLE_DEVICES empty, repository venv, vendored Transformers,
one CPU thread and a fresh basetemp under checkpoints. Exact command:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=transformers/src:src:. ./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/v5/test_throughput_attention.py tests/stage1/v5/test_selection.py tests/stage1/v5/test_execution.py tests/stage1/v5/test_replay.py --basetemp=checkpoints/stage1-v5-gpu-utilization-20261018-tests/intake-baseline > logs/stage1-v5-gpu-utilization-20261018/intake-baseline.log 2>&1
```

Observed exit0, **64passed/2skipped in12.97s**; complete output read. Log
SHA25681260e8769331b8b1b2c0f964eb374a15015eabd37e903029c72b1819bc79618.
These are baseline witnesses, not tests or measurements of a new implementation.

Additional static finding: the current eight runtime source bindings cover
Teacher/query/history and parent/v5 Trainers, but not the source modules that
implement dropout ownership, student attention capture, packed replay or text
selection. The design must explicitly bind changed execution modules/profile
so a changed implementation cannot silently present the old runtime identity.
Keep old completed contracts/checkpoints inspectable and unchanged. Do not
weaken existing strict resume comparison or claim compatible continuation
without separately reviewed evidence. No new loss/query formula is proposed.

Implementation remains unauthorized pending completed architectural design
sections, critique commitment and final written approval. Scope and numerical
policy are now confirmed. No human response is inferred from elapsed time,
and no product changes have been made.

## Architecture section A1 proposed after scope confirmation

Use the existing F -> reference -> text selection -> packed V/Text-off ->
loss/backward flow and existing public interfaces. Change four internals:

1. Replay mask: construct prefix visibility and suffix causal visibility by
   broadcast/index operations, then apply the V body and selected Text key
   exclusions. Preserve exact zero/-infinity values, padded-query handling,
   original predictor coordinates, branch/row order and attached F-prefix
   gradients. Do not allocate a second dense floating replay-sized workspace.
2. Source-coordinate dropout: retain private generator, seed, original
   source-length x hidden draw shape and scaling arithmetic; gather the masks
   with tensor indices instead of constructing one Python element per token.
   Keep per-source/site generation local; no persistent all-site mask cache.
3. Student attention observation: use bounded query chunks per sample/layer,
   with native RoPE, original positions, full visible-key FP32 softmax,
   head/layer averaging and detached observation. No candidate-only softmax
   and no full all-query attention tensor. Batching can perturb floats; it
   cannot be accepted solely on an attention tolerance if selected keys differ.
4. Text selector: stable GPU sorting with explicit original-position tie
   handling, transfer only the ranked prefix needed by the existing cap plus
   original-device masses/rhos, and retain Python-double sequential threshold
   accumulation. Preserve low-mass, zero-cap, empty, tie and boundary behavior.
   Do not replace the cumulative decision with a parallel GPU prefix sum.

Bind these execution modules in the existing runtime source identity so a new
implementation cannot silently resume under an old source contract. Preserve
historical contracts/checkpoints for read-only inspection. Do not add a new
query mode, loss coefficient, sampler, launch framework or training run.

Exact selection under batched float perturbations is not established by
static reasoning or the unchanged baseline tests. The next section must
define the comparison witnesses and numerical limits, with a mismatch blocking
activation rather than being explained away by float tolerance. Finite tests
will not be described as a proof over every future input. No GPU premise probe
has been authorized or executed for this proposal.
