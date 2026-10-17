# SDD ledger — plan: docs/superpowers/plans/2026-09-16-evidence-replacement.md

Spec: design commitment v1, SHA256 8fa8e36fb433ebe2e7563852ee309ce1ab2e97fda6f6a30bc6841ab1983e1f4e.
Approval: user reply "진헹하자"; implementation/engineering verification, no training.
Branch: feat/ours1-evidence-replacement-20260916. Baseline: ae5a1cd.
Source branch: ours1. Existing dirty source snapshotted without modifying it.

## Prerequisites

- using-superpowers/brainstorming/design hooks executed in design record.
- writing-plans read/executed; 3-task plan authored by root; no product code authored by root.
- using-git-worktrees executed: root checkout was not already linked; no native worktree tool; ignored .worktrees path; approved design called for isolation.
- New branch copied all tracked local changes and relevant untracked files; source record at results/evidence-replacement-20260916/source-snapshot.json.
- Existing venv reused; data/checkpoint/venv links retained. Baseline focused test command in logs/evidence-replacement-20260916/baseline.txt: 204 passed, 2 deselected, 161.61s.
- TDD, requesting-code-review and finishing skills read. Final full-suite command selected in plan for shared verification/finishing use once after final review. No finishing claim yet.

## Preflight task/interface scan (actual controller self-review)

| Tasks | Shared file/interface | Produced versus consumed | Finding |
|---|---|---|---|
| 1 / 2 | replacement.py records/planner | Absolute source positions, predictor IDs, source_kind/donor ID consumed by scorer | Signatures and semantics agree |
| 1 / 2 | replacement_helpers.py | Task 1 planner fixture extended with actual tiny model in Task 2 | Sequential ownership, no parallel edits |
| 1 / 3 | canonical sample/step IDs and sampling version | Planner expects sample_id/seed; Trainer passes same collator identity/training seed | No new identity construction in Trainer |
| 2 / 3 | shared inputs and results | sample_id/replacement_seed/replacement_version; visual_original/visual_replacement results | Exact strings match plan |
| 2 / 3 | Qwen dispatch versus Trainer/config | Model support lands first, public mode follows | Task 2 direct scorer/model tests independent of Task 3 config |
| 2 / 3 | helper/runtime tests | Tiny actual encoder/model fixture reused by Trainer integration | No new training jobs required |
| 1 internal | tests versus planner | Literal source indices (3,4), donor(8,9), shifted predictor rows | Hand-checked, seed fixture independently calculated |
| 2 internal | replay versus native visibility | 2-D dense/native comparison, exact causal/sliding Flex and CUDA witness | No dependency on old incorrect evidence mask |
| 3 internal | config/loss/metadata/logging | Constraints match spec; sum/count logging separate from step-mean loss | No public Stage2B resume expansion |

No task/global-constraint conflict requiring a ruling found. Plan self-review checked all spec sections, no unresolved placeholders or undefined cross-task interfaces.

## Task state

- Task 1: pending handoff.
- Task 2: pending Task 1 review.
- Task 3: pending Task 2 review.
- Final whole-feature review: pending.
- Fresh CPU suite/CUDA verification and human branch decision: pending.

Plan cold-read: /root/replacement_plan_cold, minor discovery gaps clarified in plan without behavior changes.
Task 1: dispatched /root/replacement_task1 (gpt-5.6-sol medium), BASE ae3ab4fb1a980295f85ad7213b604f5ad98eff9b. Global constraints included verbatim in brief. TDD and no-subagents contract sent.

Task 1: implementation 6e49a5e reported DONE; report read, RED expected missing module, GREEN 66 passed/2 deselected. Independent review /root/replacement_review1 (gpt-5.6-sol high), package ae3ab4f..6e49a5e, pending.

Task 1: review found 3 important validation gaps; fix round 1/5 dispatched to original implementer, FIX_BASE 6e49a5e. Cross-task cautions tracked in Tasks 2/3; not claimed complete here.

Task 1: fix round 1/5 (3 addressed, 0 open; commits 6e49a5e..25536e9).
Task 1: complete (commits ae3ab4f..25536e9, review clean).
Task 2: BASE 25536e94b8070f189308118d389703d5771ae8c7; dispatch pending.
Task 2: dispatched /root/replacement_task2 (gpt-5.6-terra high); global constraints and TDD/no-subagent/report contract provided.

Task 2: implementation de160103fef3e19e69a1b335303a7c3c9a46f308 reported DONE; report read. CPU 93 passed/1 skipped/2 deselected, actual CUDA Flex 1 passed. Independent review /root/replacement_review2 (gpt-6-astra high), package25536e9..de16010, pending.

Task 2: minor (deferred): replacement-only keys accepted in old mode validator (review2: stage2_shared_prefix.py:15).
Task 2: minor (deferred): donor-copy detach test does not isolate copy contribution (review2: runtime test:94).
Task 2: complete (commits25536e9..de16010, spec PASS and quality Approved;2 minors for final triage).
Task 3: BASE de160103fef3e19e69a1b335303a7c3c9a46f308; dispatch pending.

Task 3: dispatched /root/replacement_task3 (gpt-5.6-terra high), constraints/TDD/no-subagents/report contract sent.
Review2 supplemental probe command/output retrieval unavailable (agent thread evicted); reviewer assertion retained, not independent durable verification. Final claim-specific tests remain required.

Task 3: implementation606c854706f4a37723aa027f6c5f3a2c2bfba247. Report read:193 passed1 skipped, later dedicated Trainer5 passed. Full output logs captured. Limits (observations, not correctness doubt): config save/reload only, no PEFT save fixture; accelerator test double, no live distributed collective. Independent /root/replacement_review3(gpt-5.6-sol high), package de16010..606c854.

Task 3: minor (deferred): direct T-off false config regression missing; implemented rejection statically verified.
Task 3: evidence-report minor corrected by controller audit append with actual retained log sequence; aggregate RED command missing, not fabricated.
Task 3: complete (commits de16010..606c854, spec PASS/quality Approved, one coverage minor for final triage).

Final review dispatched /root/replacement_final_review (gpt-6-astra high), full-feature package ae5a1cd..606c854; all task/deferred/evidence limits supplied.

Final review: one Important ineffective donor-detach test,3 minor schema/target-scope/config-test findings. All4 enter the single final fix wave. No material design change; BASE606c854.

Final fix dispatched /root/replacement_final_fix(gpt-5.6-terra high), implemented3112e9677771269ddd38a1a1e2337840e66d3458. Report read. Focused86passed1skipped, schemaRED and causal detach mutation failure both preserved. Scoped rereview /root/replacement_fix_review(gpt-5.6-sol high), package606c854..3112e96, pending.

Final scoped review:3 addressed,1 open, no new breakage;606c854..3112e96.
Ruling: Park the residual automated scope-specific independent-score assertion as a real regression-coverage minor, retain it explicitly, and obtain a read-only independent literal-score runtime witness for both distinct scopes during final verification — production mapping/targets are exercised and no functional defect is alleged; another product fix wave is prohibited by SDD — if wrong, future scope-scoring regressions could evade the automated Trainer test despite current correct runtime values. Do not claim all regression coverage findings resolved.
Supplemental verification assigned /root/replacement_fix_review with exact known target rows/IDs and native independent score requirement; report pending. Full CPU suite plus CUDA remain required; no production changes after3112e96.

Final verification running on3112e9677771269ddd38a1a1e2337840e66d3458. Both H100 NVL GPUs idle at preflight; git diff --check clean.
CPU command: PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q -m 'not gpu and not distributed'; session83322; full output logs/evidence-replacement-20260916/final-cpu.log.
CUDA command: RUN_REPLACEMENT_CUDA=1 PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_replacement_runtime.py::test_cuda_flex_dispatch_matches_dense_outputs_and_gradients; session28893; full output logs/evidence-replacement-20260916/final-cuda.log. Independent CPU/GPU commands parallel, no product mutations.
Finishing gate invoked alongside verification-before-completion; reuse CPU suite for shared test-pass claim only.

Supplemental target-scope witness PASS on3112e96:8 independent literal score comparisons,all maxerror0.0; exact command/code/output read and retained. Automated scope-specific independent-score assertion remains parked minor.
Final CUDA exit0:1passed12.37s; complete output read. CPU full suite still running, one observed failure awaiting traceback; no completion claim.

Final suite FAIL exit1:6failed3297passed6skipped5deselected11warnings849.83s. Includes two unmarked old CUDA tests despite marker exclusion, so not a pure CPU-only run. No finishing menu permitted. Systematic-debugging activated. One new save fixture/initialization incompatibility assigned for scoped verification repair; remaining5 assigned baseline comparison, no inference of pre-existence yet. This repair responds to new runtime verification failure, not another final-review fix wave.

Verification repair dispatched /root/replacement_verification_repair(gpt-5.6-sol high), BASE3112e96. Baseline diagnosis fresh spawn unavailable (thread limit); reused /root/replacement_task2(gpt-5.6-terra high) for read-only prefeature runtime comparison, not an independent code review or product implementation.

Verification repair4b27265907a595278855b5382a78d263171ecc84: onlyfixture+report, RED1failure/GREEN87passed; independent review /root/replacement_causal_lens specPASSqualityPASS. Fresh focused completion verification running at4b27265. Baselineae5a1cd comparison reproduced3 existing failures;2 old CUDA cases passed baseline and current-order investigation pending. No full-suite pass claim/menu.

Final focused verification at4b27265:141passed1skipped12.86s exit0; command uses config,replacement_plan,replacement_runtime,replacement_trainer,stage2b_trainer,shared_prefix_model with -m 'not gpu and not distributed'; complete log final-focused-after-repair.log read. Final new-mode CUDA at4b27265:1passed12.15s exit0, final-cuda-after-repair.log read.
Baseline disposition:3 non-CUDA failures reproduce on ae5a1cd.2 old CUDA failures pass isolated on baseline/current; forcing bothTF32 flagsTrue reproduces2fail on BOTH baseline/current. Original suite state not directly captured; order/globalTF32 explanation is inference. No unrelated fixes.
Finishing Step1 blocked by ungreen full suite; do not present integration menu or claim overall completion. Source ours1 diff and bound-design hash freshly unchanged. Feature implementation committed; no training/merge/push. Parked regression minor remains explicit. Scoped SDD workspace retained (not cleanly closed); durable record copies below.
