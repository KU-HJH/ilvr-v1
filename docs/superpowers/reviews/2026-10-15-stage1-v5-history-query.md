# Stage1-v5 history query: independent implementation review evidence

Approved spec: `../specs/2026-10-15-stage1-v5-history-query-design.md` at
9c402ff2daa49c1e293b4eb0829f3402389f2839. Approved plan:
`../plans/2026-10-15-stage1-v5-history-query.md` at
8e6f81751c66d410c3e882a594fa14b49a3869d6. User selected Subagent-driven.
This record retains minimal actual reviewer evidence; task diffs are available
from the exact Git ranges below. Design advisory reads are not code review.

## Task1 — config/layout/math

Implementer `/root/history_task1_impl`; independent reviewer
`/root/history_task1_review`. Range:
8e6f81751c66d410c3e882a594fa14b49a3869d6..
e38e21c881b3d9a28d01899d48f90a56229a1e14.

Returned verdict: **Spec compliant within Task1; Task quality Approved.**
Critical/Important/Minor findings: none. No reviewer test reruns.

Reviewer evidence:

- Public config, strict saved schema, segmentation and FP32 math:
  `src/stage1/query.py:14`, `:91`, `:141`, `:191`.
- Resume conflict/malformed semantic rejection: query.py:91.
- Complete latent-block, special/mask exclusion and no final tail: query.py:141.
- Small mass and query denominator/finite/norm checks: query.py:215.
- Real numerical/rejection tests without mocks: test_history_query.py:14;
  final focused log:83 passed, no warnings.
- Named unchanged-type risk checked: Stage1LatentSpan at stage1/types.py:99
  describes complete latent blocks. Guessed collator module was absent;
  reviewer did not claim collator runtime compatibility.

Cannot-verify items and root resolution: actual DoRA/native equivalence belongs
to Tasks2/3; collator/injected targets to Task4; model topology to Task5;
whole-project regression to final gate. None is claimed by Task1. Intermediate
broad run was interrupted after628 passes (exit2), not a full-suite pass.

Root read report and complete final focused/interrupted logs:
`logs/stage1-v5-history-query-implementation/task-1-final-focused.log`
(83 passed in2.79s), `task-1-full-suite.log` (628 passed in252.20s then
KeyboardInterrupt). RED logs preserve83 absent-API failures before production
implementation. Task1 is complete at the reviewed commit; no fixes required.

## Task2 — effective DoRA prefix adapter

Implementer `/root/history_task2_impl`; reviewer `/root/history_task2_review`.
Range e38e21c881b3d9a28d01899d48f90a56229a1e14..
2034445b0657f40d4cf59c0204426e7bb0ffc6fe.
Returned verdict: **Spec compliant; Task quality Approved.**
Critical/Important/Minor: none. No reviewer tests or edits.

Reviewer evidence: frozen records/interface/backbone/image preparation,
injection/MRoPE/forward at history_teacher.py:9,39,62,137,173,178; optional
diagnostics/no-op validation at :147 and test_history_teacher.py:207,217.
Independent native PEFT oracle at tests:41,52,67,117; single encode/no head,
RNG/cache/grad/storage observations at tests:102,124,134,148. Named wrapper-
bypass risk traced through peft_common.py:15 and custom modeling :1497,1616,
1969,2896,2915,2938. Non-generation/normalized hidden semantics agree.

Final focused log:28 passed in3.55s. Four actual native FP32 CPU cases have
zero embedding/position/mask/hidden error, with nonzero effective DoRA checked
against plain projection. Root read complete final/native logs and comparison
JSON under logs/results/stage1-v5-history-query-implementation/. Native,
diagnostic-empty and empty-no-op RED failures are retained there.

Cannot-verify resolution: Task4 must enforce single assistant, original images
before assistant, helper candidates and current START as stop (adapter accepts
other complete prefix boundaries). Task3 owns the observer, Task5 resume;
GPU/BF16/vision FA2/four-rank/real-length evidence remains unverified. No broad
runtime claim is inferred. Task2 completes at the reviewed commit.

## Task3 — reduced attention, initial review

Implementer /root/history_task3_impl; independent reviewer
/root/history_task3_review (gpt-6-astra, high). Initial range2034445..7d291f5.
Returned Spec issues / Task quality Needs fixes, one Important and one Minor.
Combined CPU gate44 passed in4.56s; nine native FP32 comparisons maxabs
7.45058059692e-09. Those short-prefix passes did not prove native mask
equivalence across the configured sliding window.

Important: history_attention.py:61 checks use_sliding_window, but native
modeling_qwen2_5_vl.py:1767 applies non-None sliding_window independently.
Reviewer focused CPU probes: window4/use_sliding_windowFalse/query18,
latent14,15/zero-logit native[0,.25] versus observer[.052631583,.052631583];
ordinary unpadded SDPA allowed only keys15,16,17,18. No existing suite rerun
or file edits. Root routed a causal regression/correction to the original
implementer. Approved full-causal scope is enforced by rejection, without
changing teacher/native execution; compatible long-prefix support is not
claimed. Fix/re-review pending.

Minor deferred for whole-change triage: attention test fixture has only one
KV head, so group-index mistakes cannot be distinguished (test:44).
Other evidence: effective projections/native MRoPE at observer:70,102,131;
streaming denominator/mass at:143; cleanup/noninterference tests:75,149.
Remaining CPU-versus-BF16/GPU/real-config/cost boundaries remain explicit.

Task3 correction ac2a8bf96588cfa824d4826a8b6524b573df3a15 was independently
re-reviewed by /root/history_task3_fix_review (gpt-6.1-sol, high): Important
finding ADDRESSED, no new breakage/out-of-scope observations. Actual native
mask boundary checked at modeling:1767-1774; guard observer:69-73 rejects
S>window before hooks, accepts equality/None, no execution mutation. Regression
tests:178-236 witness native masks/probabilities and rejection/boundary behavior.
RED2/expanded8 failures followed by GREEN30 passed in4.64s; complete log read.
No re-review test reruns. Task3 completes, with the Minor left for final triage.

## Task4 — adaptive builder and legacy parity

Range ac2a8bf96588cfa824d4826a8b6524b573df3a15..
8d729c992254f626c26c33cc5720230e3c2e4b68. Implementer
/root/history_task4_impl; independent /root/history_task4_review
(gpt-6-astra, high). Spec compliant / Task quality Approved; no findings.

Reviewer traced dispatch/unchanged legacy at targets.py:231,254,390;
debug-off trajectory/image/mask/helper validation at:124,406,425,465;
pre-forward max-prefix bound at:476; prompt/ledger/contextual mass/injection
at:500,511,514,528,539. Dependency risks checked in query.py, history_teacher,
inherited intent/header/pooling and observer empty/cleanup paths. Tests:63,90,
117,145,178,274,290,322,366,427 supply comparative/native/synthetic witnesses.
No reviewer test reruns or edits. Workflow interpreter deviation remains
recorded; these later checks do not retroactively establish compliance.

Root/reviewer read complete logs/stage1-history-task4-20261015/verification.log:
65 passed in3.92s, pristine. Task RED28 absent-interface failures and later
RED3 no-latent structural validation failures precede their fixes. Runtime
metadata results/stage1-history-task4-20261015/runtime-witness.json records
exact previous/all legacy targets/masks/indices/calls/RNG, adaptive prefix
lengths11,13,19, exact prior raw-target injection, untouched inactive pad and
empty optional diagnostics. Independent native math and known synthetic
dependency distinguish contextual teacher state from raw-vector pooling;
future/current-helper query isolation and repeated-build tests pass.

Cannot-verify disposition: Task5 wiring, Task6 parent-Trainer integration,
final full suite remain required. Real-checkpoint/GPU/BF16/FA2/four-rank
runtime and long-input/performance remain unverified. Task4 complete.

## Task5 — CLI, contracts, resume and existing shells

Initial range8d729c992254f626c26c33cc5720230e3c2e4b68..
c073b8da83df639a2f922c60c79b7e9d18cb6ba1. Implementer
/root/history_task5_impl; /root/history_task5_review (gpt-6-astra, high).
Verdict Spec discrepancy / quality Approved with Minor; no Critical/Important.

Strict-schema gap: v5_resume.py:36,120 treats explicit query_config:null as
absent historical query metadata. Reviewer focused read-only probe confirmed
early legacy acceptance, but actual runtime comparison rejects null versus
the saved legacy object; no authentication bypass. Root confirmed the spec
gap and routed a targeted original-implementer null regression/correction.
Fix and scoped re-review pending. Existing source-bound null checks remain.

Verified reviewed paths: parser explicitness/restore-before-conflicts at
utils_deepseed.py:383-495; composition:520,728 and trainer:283 plumbing;
runtime query/source/loss/reference/topology/layer validation at v5_trainer.py:
264-332; worker omission:427, recipe sourced-state clearing:76. Named outside
checks: query.py:92-126, multinode.py:1910-2008, multinode_runtime.py:895-933,
adaptive profile at targets.py:403-411. Read truncated-hunk context as needed.

Root/reviewer read pristine logs under
logs/stage1-v5-history-query-implementation/task5/: final-focused.log215
passed in15.77s, final-composition-fixture.log8 passed in10.65s. Source-key
regression passed in prior combined gate, shell syntax checks exit0.
Results/task5/both-node-argv.json (under the matching implementation result
directory) contains six actual coordinator→mapped-worker→Python argv records:
fresh legacy, adaptive, and omitted-query adaptive resume gamma.55/last4 on
rank0/rank1, configured four ranks/two machines. No actual remote execution,
GPU or distributed resume claim. Logged suites not rerun by reviewer.
Flat task5 log storage was corrected byte-preserving to the run subdirectory.

Correction8fe8717cba98eac6af5d23a5fd6799900b9f185f: present nonmapping
query metadata now rejects at v5_resume.py:36-39; historical fallback requires
key absence at:122-130. Four build/load/loss/query regressions:189-208 failed
RED then resume file GREEN36 passed in9.01s (task5/fix-round1-green.log).
Independent /root/history_task5_fix_review (gpt-6.1-sol, high): ADDRESSED,
no new breakage/out-of-scope findings, no reruns. Task5 complete at8fe8717.

## Task6 — actual parent-Trainer CPU witnesses and README

Range8fe8717cba98eac6af5d23a5fd6799900b9f185f..
f93e9eebc7584eb2dfa84ed575910331134451f3. Implementer
/root/history_task6_impl; independent /root/history_task6_review
(gpt-6.1-sol, high): Spec compliant within CPU scope / quality Approved,
no findings. Read-only, no rerun. Actual parent dispatch/backward/replay traced
at test_history_integration.py:29,69 and v5_trainer.py:102,117,159;
shared snapshot/attached trace:97, literal GAS/native CE:74, grad/reference:118,
calls:89, A/B/A and saved query:134,157. Shared-fixture default paths/packed
keyword checked at test_trainer.py:142,179,250,281,416 and replay.py:85.

Complete final log task6/final-focused.log:8 passed in10.24s, six actual runtime
JSON records, no warnings. Approved test-only adaptive removal:3 expected
call-sequence failures, restored bytes then GREEN. Active M2 calls adaptive6,
legacy4; partial1/2-slot and full4-slot windows keep denominator4 and original
block-count auxiliary scaling including empty slots. F/reference target storage
shared, packed trace attached with exact target rows; student LoRA gradients,
optimizer step and frozen reference digest/RNG/hook invariants witnessed.
Saved adaptive .55/all config through real parser reproduces next targets under
the identical frozen teacher; this is not full checkpoint state restoration.

README:128 adds bounded Training subsection; :178 states exact native window
bound. README preview and actual emitted coordinator --dry-run exit0 with
configured2nodes/fourranks; task6/readme-dry-run.log and matching results
contain evidence, no launch. Reviewer checked Task5 composition uses substituted
Trainer/distributed interfaces while Task6 bypasses __init__ and exercises
actual parent backward. These are complementary witnesses, not a fully
authenticated production training/resume execution. Production initialization,
GPU/BF16/FA2/four-rank/full restore/long-input/performance remain unverified.
Task6 complete; whole-change review and final suite remain required.
