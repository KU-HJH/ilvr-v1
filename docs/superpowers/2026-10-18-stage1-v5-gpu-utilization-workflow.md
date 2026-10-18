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

The second, pending scope question presents three approaches: optimize the
four operator paths while retaining sampler/rank assignment (recommended),
also rebalance ranks within each existing update, or start only with mask and
dropout. The requested rank-improvement direction is not silently discarded;
the decomposition and its first deliverable need confirmation.

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
3. Clarifications: numerical policy answered; scope/decomposition pending.
4. Alternatives: three scope approaches presented in the pending question.
5. Architecture/data-flow/error/testing section confirmations: pending.
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
