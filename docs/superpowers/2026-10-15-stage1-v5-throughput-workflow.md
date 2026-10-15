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

## Verification claims and required witnesses

- Discrete selection is unchanged: frozen-original selector oracle across threshold,
  tie, empty, low-mass and multi-sample cases; exact selected key comparison.
- Observation arithmetic is unchanged: actual Qwen trace/native attention and
  original operator comparisons, identical matrix shapes/reduction order; attached
  replay KV and detached observation behavior.
- Training graph remains supported on tested fixtures: existing actual tiny Qwen
  loss/gradient/dropout/checkpoint tests. This does not prove full-size BF16 or
  four-rank equivalence.
- Collation semantics are unchanged: full prepared tensor/metadata comparison with
  old three-call implementation and the local original Qwen processor, including
  no-helper and multiple images with variable grids; malformed cardinality rejection.
- Redundancy is removed: causal operation/transfer counts and image/processor-call
  counts; same-fixture before/after local measurements including memory.
- Project regression suite: final command after independent reviews is
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q`.
  `pytest.ini` collects `tests/`. CUDA remains visible. This full-suite run is shared
  evidence for verification-before-completion and finishing Step 1, run once on the
  final reviewed product revision. Additional timing is a distinct performance
  witness; no full training or GPU-utilization claim is made.

Task 1 implementer: `/root/throughput_attention_impl`, gpt-6.1-sol/high,
BASE `bb942b7323495cda716b7c5b889d6a8dc699aa23`. Work is task-scoped and sequential;
root owns coordination, not product implementation. Task 2's real processor source
is existing `checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct`; no download.

## Task 1 result and independent gate

Implementer `/root/throughput_attention_impl` committed
`73dca0f531187ad28ed6fe7fcd0ea52bf908e39d` (BASE bb942b7). Changes: per-sample/layer
FP32 K reuse; invariant coordinate/index reuse; compact candidate handoff; bulk
weights/mass/rho read retaining original-device FP32 reductions and Python selection.
Frozen-original source hashes: trace
`1a7d4bfb8b1d968e0a40541642b78e8fa5dbf61d4570af4c8effe3fe42441724`; selector
`3530678a6f9f7673d91d8305c2895e5904ca648f2dbbb8d34019e446e3f99827`.
Causal RED logs record K copies 24 vs required 4, scalar reads 16 vs 0, candidate
index constructions 24 vs 2; mixed-device direct-call regression also passed
after RED. Final focused command used the standard environment above and
`-m pytest -q tests/stage1/v5/test_throughput_attention.py tests/stage1/v5/test_selection.py tests/stage1/v5/test_execution.py tests/stage1/v5/test_replay.py tests/stage1/v5/test_debug.py tests/stage1/v5/test_trainer.py`:
**119 passed in 16.38s**, no warnings, log `task1-final-verification.log` under
`logs/stage1-v5-throughput-20261015/` (root read full output).

Independent reviewer `/root/throughput_attention_review` (gpt-6.1-sol/high)
reviewed bb942b7..73dca0f: spec compliant, quality Approved, zero Critical/Important.
Verified frozen blob IDs against base, actual call consumers, named native Qwen
assertions and raw RED/GREEN/measurement evidence without rerunning the suite.
Minor deferred for final review: grouping interleaved samples changes trace
mapping insertion order; consumers found use keyed lookup and no documented
ordering contract. No observed output/gradient impact; this remains an unresolved
minor observation, not erased by approval. Full-size/distributed claims remain
unverified and out of scope; final full-project suite remains pending.

Final local benchmark reproducible with `scripts/benchmark_stage1_v5_attention_selection.py --device cuda:0 --output <results path>` (or cpu), standard environment.
JSON/log stems `task1-cuda-candidates-final` and `task1-cpu-candidates-final` under
this task's results/logs directories. H100 NVL, 3 warmups, 20 repetitions,
BF16-projection/FP32 observation fixture, batch2/length512/64queries/2layers;
selector 32queries/128candidates/row4096/vocab17, validation=False. Exact original
attention and selected keys on CPU/CUDA. CUDA medians ms: attention16.134232→10.771047,
selection2.412995→1.042174; K copies128→4, index constructions128→2, scalar reads32→0,
list reads34→1. CUDA extra peak bytes: attention5804032→5811200;
selection12800→41472. Python packing peak59606→193428B. These buffers are bounded;
no memory reduction or training speedup claimed. Initial CUDA benchmark had an
incorrect full-vs-compact argument; failed evidence and causal regression are
preserved. This measurement bug is separate from the design hate pass's first
nail, which remains unrun (no end-to-end phase profile).

Task 2 implementer `/root/throughput_collator_impl` (gpt-6.1-sol/high), BASE
73dca0f. Causal RED: 9 processor calls vs required3 across three samples.
Initial focused result82passed/1failed: existing SourceProcessor fake generates
whole-call arange pixels, so the same helper changes values depending on preceding
originals; actual Qwen per-image equivalence passes. Root ruling: allow the smallest
image-content-dependent fixture correction, retain metadata assertions and failed
log, require independent review of it. Cost if wrong: could hide a real ownership
regression; frozen-original and actual processor comparisons are required evidence.

## Task 2 implementation evidence

Commit `e5a1c1c44c458cdaef35063ff877a59d80c4057f` (BASE73dca0f), six scoped files.
V5 full-conversation processor output is split by original-image grid count and
sum of raw grid products; role validations and all downstream token/metadata
paths remain. Standard retains three calls; no-helper retains `(0,0)` default-dtype
pixels. Frozen original complete data.py SHA256
`d132b34f4a2a79403b7f850b9d667a4e947bf368a807da31856591fb81e2a183` matches BASE.
Causal RED exact command: standard environment, `-m pytest -q tests/stage1/v5/test_throughput_collation.py::test_v5_materializes_each_image_once_and_processes_each_sample_once`,
exit1 (`assert 9 == 3`), `task2-red.log`. Final focused command:
`-m pytest -q tests/stage1/test_data.py tests/stage1/v5/test_metadata.py tests/stage1/v5/test_throughput_collation.py tests/stage1/v5/test_execution.py` with standard environment,
**83passed/4.90s**, no warnings; `task2-green-final.log`.

Final offline witness command uses standard environment plus
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and
`-m tests.stage1.v5.benchmark_collation --processor-path checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct --output results/stage1-v5-throughput-20261015/task2-collation-default.json`.
Exit0; log `task2-qwen-benchmark-default.log`. Production-default loader yields
Qwen2_5_VLProcessor/Qwen2TokenizerFast/saved Qwen2VLImageProcessor; expected slow
image processor advisory occurs3times, no errors. Eight complete exact-output
comparisons cover ordinary/occurrence singleton and batch; input IDs, masks,
labels, pixels, grids and both metadata objects agree with frozen original.
Fixture3samples/4originals/3helpers includes variable sizes and no-helper;
raw original/helper rows1792/2176, patchwidth1176. Generated images remain under
data/stage1-v5-throughput-20261015/collation-witness. Separate fresh processes per
mode,2warmups/15repetitions, full collation median64.632654→34.681149ms. Calls9→3,
opens14→7. Tracemalloc peak32383269→26419544B; whole-process VmHWM1001552→1007540KiB.
RSS includes imports/native allocator and does not demonstrate memory reduction.
Small synthetic inputs and warm file cache support local savings only.

Preserved attempts: explicit use_fast=False chose an unsuitable non-offset
Python tokenizer and failed in the old oracle; explicit use_fast=True selected
an alternate image backend and passed, but final witness above uses production
defaults. Preliminary mutable-count-list and RSS-reporting issues were corrected
in measurement code, with earlier logs retained. No product fallback or tolerance
relaxation was introduced. Independent reviewer `/root/throughput_collator_review`
(gpt-6.1-sol/high) now has brief, report, exact diff and fixture-correction evidence.

Task 2 independent verdict: spec compliant; quality Approved; zero Critical/Important.
Reviewer verified frozen blob `a2c7a1e9166e8df5faabe1f49652e4bba84bcf8c` against base,
actual Qwen per-image/raw-patch implementation, unchanged existing metadata
assertions, causal RED and83-pass GREEN, eight exact comparisons and measurement
outputs. No tests rerun. Minor deferred: expected slow-image-processor advisory
appears3times; report already classifies it explicitly, final reviewer will triage.
Model-level behavior and end-to-end speed are not inferred universally from the
collator comparison. Task 1 tiny-model checks and final full-suite gate are separate.

Both task gates passed. Final broad review covers all new changes since intake
148b0e8 on the existing branch, including interactions between the two changes,
with the earlier reviewed Stage1-v5 revision as baseline. Deferred minors: trace
mapping insertion order and expected processor advisory. No parked finding or
process violation. Final verification and finishing have not yet executed.

## Final independent review and completion gate

`/root/throughput_final_review` (gpt-6-astra/high) reviewed148b0e8..61c07a6:
Ready to merge conditional on fresh full-project tests; zero Critical/Important.
Verified byte-identical originals, actual runtime boundaries, causal logs,
119/83 focused results and recomputed final medians from raw repetitions.
Fixture ruling accepted based on actual per-image Qwen implementation and
independent real-pixel/full-processor evidence. Advisory is expected/no defect.
Trace insertion-order observation remains a disclosed nonblocking minor;
known consumers are keyed, with no ordering guarantee. No fix wave needed.

Controller dispositions for every declined item: production-size BF16/four-rank
and B200 claims remain out of scope; end-to-end speed/utilization/quality requires
separate authorized execution; arbitrary batch-dependent/stochastic processors
are outside the existing Qwen contract; unchanged baseline review is reused with
integration checked here; fresh full-suite success is a pending required gate.
No uncertainty is silently promoted to equivalence or waived by review.

Verification and finishing are now active. Root will run fresh distinct local
performance witnesses first (no contention), then the recorded full project
command once for both verification and finishing Step1. This review does not
establish final test success. Host worktree and branch integration are human-owned.

Fresh root performance witnesses on the reviewed product content both exited0:
`root-attention-cuda.json/.log` and `root-collation-default.json/.log`, with the
same commands/fixtures above and new output names. Complete logged JSON parsed,
exact-equality flags checked and medians recomputed. Attention15.984324→10.700707ms;
selection2.4148565→1.0417705ms; collation67.494156→37.245753ms (1.81213x).
Counts and exact comparisons unchanged. CUDA extra peaks as in Task1 report;
collation tracemalloc32382669→26419152B, VmHWM1001428→1001820KiB. Only expected
saved-slow-image advisory appears in collation preamble. No test/benchmark
concurrency or training launch. These fresh measurements substantiate local
path claims, not utilization/end-to-end claims.

## Fresh final verification result and handoff

Full project command above ran once at `91bc90f96b01172e0b357deffe0e41421aed2837`:
**4642 passed, 11 skipped, 1 warning in1207.69s**, exit0. Root read the complete
`logs/stage1-v5-throughput-20261015/root-full-suite.log`. Warning is
`TORCH_CUDA_ARCH_LIST is not set` from existing
`tests/stage2/test_full_optimizer_offload.py::test_installed_cpu_adam_mapping_preserves_no_decay_group`;
no failures. Skips are reported as skips, not passes. This single result supplies
both verification-before-completion and finishing Step1's overlapping test-pass
claim. No duplicate full-suite run. Distinct local performance commands are recorded
above and passed on the same product source content.

Post-suite diff checks passed: whitespace for148b0e8..HEAD; exact empty diff
61c07a6..HEAD for src/transformers/scripts/tests confirms final reviewed content
is the tested content. Only evidence docs changed after final review. All source
and test edits are committed; unrelated pre-existing caches and cancelled retrieval
workflow remain untouched. No model artifacts, training, remote synchronization,
merge, push, environment changes or profile changes occurred.

All approved implementation requirements are complete with independent task and
final review and fresh verification. Remaining known minor: interleaved trace map
insertion order changes, without an observed ordering-dependent consumer. Bulk
packing can increase temporary memory. Neither is silently represented as universal
compatibility/memory savings. Full training speed/utilization/BF16 four-rank behavior
was not measured and is not claimed.

Finishing Step2 verified named Stage1-v5 branch in host-managed linked worktree.
Branch origin9119b55 matches existing dil-bce-orig; any integration remains the
human's choice. Worktree is preserved. The final response presents the three
finishing options after this verified result; no integration decision is inferred.

Durable detailed evidence under `results/stage1-v5-throughput-20261015/`:
`task-1-report.md`, `task-2-report.md`, `task-1-review.md`, `task-2-review.md`,
`final-review.md`, `workflow-ledger.md`. These preserve agent identities, commands,
outputs, failures/fixes, review verdicts and the root fixture ruling. Frozen oracles,
tests, measurement scripts and diffs live in Git. Only this task's owned SDD scratch
is removed after preserving those records; other scratch/worktrees remain.

Actual lifecycle: readchk scope resolved; systematic-debugging investigation;
bounded brainstorming with frozen v1 hate/macrothink and user `진행해` approval;
writing-plans nonapplicable (bounded task brief); using-git-worktrees verified
existing isolation; SDD sequential implementers with TDD and independent per-task
spec/quality review; requesting-code-review final broad review; fresh verification
and finishing Step1–3 executed. No workflow gate was substituted by self-review.
Integration menu is the remaining human-owned handoff, not unfinished implementation.
