# Stage1-v5 throughput optimization workflow

## Intake and current authority

Understood as: reduce the two identified sources of Stage1-v5 training overhead:
Text-off attention observation/selection and synchronous repeated CPU input
preparation, while preserving the existing training objective and data semantics.

User request: `1번, 2번은 해결하자` following the performance investigation.
User selected subagent implementation with root coordination/verification on
2026-10-15. This selects the implementation method; it does not approve the
design below. Subsequently the user said `진행해`, approving design commitment v1
and its stated local-performance / end-to-end attribution uncertainty. Product
implementation within that exact scope is now authorized; no full-model training
or costly premise experiment is authorized.

Workspace: `/root/.codex/worktrees/6b9d/ILVR`, branch `Stage1-v5`.
Intake HEAD: `148b0e8ead14456d02cc595f0cc32a0752ee22e8`.
No tracked product changes at intake. Existing untracked Python caches and the
cancelled retrieval workflow belong to earlier work and remain untouched.

## Lifecycle

Active owner: Superpowers. Active phase: subagent-driven implementation, bounded existing-path
optimization. Existing public interfaces, training profile, sampler and objective
remain fixed. No new subsystem or implementation-plan document is proposed.

- using-superpowers: previously read and applied in this conversation.
- systematic-debugging: read and applied to the preceding read-only investigation;
  hotspot candidates are identified, but no stage-attributed runtime profile exists.
- readchk: executed by root; numbered scope resolved from the preceding answer.
- brainstorming: context explored; intended outcome and preservation constraints
  stated; implementation method selected; final design v1 approved by `진행해`.
- hate: executed by root against frozen proposal v1; exact result recorded below.
- macrothink: required at this commitment because optimizing unmeasured overhead
  can consume substantial implementation/validation work without fixing the dip;
  three bounded same-model advisory reads completed; classification recorded below.
- After exact design approval: concrete task brief, subagent-driven development,
  applicable TDD, independent specification-and-quality review, fresh verification
  and applicable branch-finishing handoff. None has executed for this change yet.

## Existing evidence

`results/stage1-v5-utilization-inspect-20261015/timing-summary.json` records two
648-update runs. First-step-excluded optimizer-step timing correlation is
0.9980366417803441, median 20 seconds, maximum 136/137 seconds. This is whole-step
timing derived from progress logs, not GPU-utilization or stage attribution.

The relevant collator, selector, trace/replay, trainer and composition files have
no diff between trained revision `c233aaf` and intake HEAD. Source tracing shows:
per-query repeated FP32 K conversion in attention observation; per-query device
scalar/list reads in Text-off selection; and three processor calls per singleton
sample with duplicate image materialization. DataLoader workers remain zero.

## Frozen design proposal v1 (not approved)

Goal: remove redundant work and device-to-host synchronization in the two named
paths without changing the chosen Text-off keys, sample order, prepared tensors,
loss formulas, gradients, helper selection, dropout or checkpoint semantics.

1. Attention/selection: keep the existing per-query attention matrix operation and
   FP32 reduction order. Hoist repeated per-sample K conversion and invariant
   coordinate/mask construction out of the query loop. Collect candidate weights,
   GPU-computed masses and rho values into bounded bulk host transfers; retain
   Python's existing ordering, tie breaking, threshold/cap/low-mass decisions.
   Do not change attention GEMM shapes, implement approximate attention, or remove
   Text-off observation. Primary files: stage1_v5_trace.py, v5_selection.py and
   their callers only if needed to preserve compact candidate ownership.
2. CPU preparation: for Stage1-v5 only, reuse the full singleton processor result
   and split its original/helper image grids and raw patch rows by image ownership.
   Materialize/process each image once. Preserve token IDs, labels, masks, image
   order, pixel values, metadata and validation failures. Keep the standard Stage1
   path and the current workers=0, sampler/GAS/resume profile. No persistent cache,
   prefetch worker subsystem, dataset rewrite or training-profile change.
3. Verification: compare against the unchanged implementation as an independent
   oracle, including real local processor inputs, multiple original/helper images,
   no-helper rows, variable image sizes, empty candidates, ties and selection
   thresholds. Require identical discrete selection and prepared tensors; compare
   actual Qwen logits/loss/gradients and existing replay/checkpoint/dropout tests.
   Measure CPU collation and the attention/selection paths separately on identical
   bounded fixtures; report memory and timing without claiming full-run speedup.
   Any full-model training/benchmark launch requires separate explicit scope and
   the mandatory two-node/four-rank coordinator, defaults and preflight.
4. Acceptance: no output/selection drift; no additional peak-memory scaling beyond
   bounded temporary buffers; demonstrated reduction of redundant processing and
   synchronization plus measured local-path timing. No claim that all utilization
   dips are resolved without an end-to-end runtime profile.

Remaining uncertainty: neither hotspot's fraction of actual training wall time
has been measured. The design deliberately avoids numerical batching and worker
changes. A substantial change to this proposal requires a new commitment version
and renewed human approval.

## Root hate pass on frozen proposal v1

```json
{
  "root": "The design assumes redundant conversion, host synchronization and duplicate collation materially cause the observed utilization dips. Whole-step timing does not establish that attribution; preserving the expensive per-query attention loop could leave the dominant cost intact even after both edits pass equivalence checks.",
  "first_nail": "Before claiming a utilization fix, collect a bounded baseline phase timing for representative short and long real inputs, separating collation, attention observation/selection and the remaining forward/backward/rank wait. If the proposed removable work is a small fraction, reject the performance premise before expanding implementation. This is a proposed premise experiment, not authorization or observed evidence."
}
```

First nail status: not run; no costly or implementation-like premise probe has
been authorized. Remaining attribution uncertainty must be surfaced in the exact
design commitment. This critique does not authorize implementation.

## Macrothink pass on frozen proposal v1

Stripped prompt: improve the two requested existing training paths while retaining
objective, selected keys, prepared tensors/ownership, dropout, gradient, schedule
and resume semantics. Known evidence is reproducible whole-step timing, not stage
attribution. Readers were asked to form an independent source-grounded view before
reading frozen proposal v1, without implementing or running probes.

Agents and outcomes (read-only; no implementation authority):

- `/root/throughput_fresh_a`: convergent. Independent recommendation was invariant
  hoisting, bulk host reads and one-pass image processing with exact comparisons.
  Compatible addition: future attribution must time the entire accumulation-window
  fetch and every rank, not one collator invocation. No incompatible finding.
- `/root/throughput_fresh_b`: convergent with compatible evidence-boundary finding.
  Preserve GPU mass reductions; compact candidate ownership avoids re-expanding
  full rows. Small-model/operator witnesses do not prove full-size BF16/four-rank
  gradient equivalence. No incompatible finding.
- `/root/throughput_fresh_c`: convergent with compatible edge-case finding, but NOT
  a fully blinded independent read: initial broad search exposed proposal snippets.
  The reader disclosed this before its final report. No-helper old pixels have
  shape `(0,0)`, unlike a naive `(0,patch_width)` slice; preserve shape and dtype.
  Keep hoisted K lifetime bounded to one sample/layer. No incompatible finding.

Root classification: no divergent-incompatible result. Compatible divergence
clusters around the distinction between local savings and full training critical
path attribution, plus explicit empty-input/temporary-memory requirements. The
two uncontaminated convergent reads are reassurance only, not correctness or
performance evidence. The third read's source findings are retained with its
exposure limitation. Root, not these readers, synthesizes the commitment.

## Design commitment v1 — approved by `진행해`

Agreed problem/scope: reduce redundant work in the two named existing Stage1-v5
paths. User selected subagent implementation and independent review. Full training
relaunch, worker/profile changes and a promise to remove all GPU-utilization dips
are outside this commitment.

Exact in-chat commitment to present:

1. Text-off 경로: attention의 기존 GEMM shape·FP32 계산 순서를 유지하면서,
   반복 K 변환과 불변 좌표 생성을 밖으로 옮깁니다. 후보 weight·GPU에서 계산한
   mass·rho는 묶어서 CPU로 보내고 기존 정렬·동률·threshold 규칙을 유지합니다.
   주요 변경 파일은 `stage1_v5_trace.py`, `v5_selection.py`이며 필요한 경우
   compact 후보 전달부인 `v5_trainer.py`까지 포함합니다.
2. 입력 준비: v5에서 샘플당 processor를 한 번 호출하고, 그 결과를 image grid의
   원본/helper 소유권과 raw patch 수에 따라 나눠 씁니다. 중복 이미지 읽기와
   전처리를 제거합니다. workers=0과 기존 sampler/GAS/resume 설정은 유지합니다.
   주요 변경 파일은 `src/stage1/data.py`입니다.
3. 검증: 기존 구현을 비교 기준으로 입력 tensor·선택 key의 정확한 일치와 경계
   조건을 확인하고, 작은 실제 Qwen 모델의 loss/gradient·dropout·checkpoint
   회귀를 검증합니다. 원본과 변경 경로의 국소 시간·메모리를 같은 조건에서
   측정합니다. full-size BF16/4-rank 동등성이나 전체 학습 속도 향상은 별도
   실제 실행 전까지 주장하지 않습니다.

Changes from proposal: made the no-helper `(0,0)` shape/dtype requirement, bounded
K lifetime, unchanged full-model evidence boundary, and accumulation-window/rank
attribution limitation explicit. No algorithm or new public interface added.

First-nail status: not run, no phase profile exists. Remaining uncertainty to
surface for explicit acceptance: these changes can remove source-observed redundant
work, but their contribution to observed GPU dips and end-to-end speedup remains
unmeasured. The subsequent `진행해` authorizes implementation of this version and
accepts that remaining uncertainty. It does not authorize a costly premise probe
or training launch.

## Implementation handoff

Bounded task brief: `docs/superpowers/briefs/2026-10-15-stage1-v5-throughput.md`.
No full implementation-plan document is applicable to this bounded path.
Existing linked worktree verified: git-dir `/root/workspace/ILVR/.git/worktrees/ILVR1`,
common-dir `/root/workspace/ILVR/.git`; named branch Stage1-v5. The host-owned
worktree is retained. Existing environment and storage links are reused.
SDD, using-git-worktrees, TDD, requesting-code-review, verification-before-completion,
and finishing-a-development-branch were read. Implementation/review/final gates
remain pending until evidence is recorded below. Root owns control artifacts only.

Focused pre-implementation baseline: selection, execution, data and v5 metadata tests: **80 passed in 4.80s**, exit 0. Exact command and log are in the SDD ledger (`.superpowers/sdd/2026-10-15-stage1-v5-throughput/progress.md`); durable log `logs/stage1-v5-throughput-20261015/baseline.log`. Preflight table covers both tasks and shared batch-metadata interface; no conflict found.
