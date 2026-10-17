# Evidence replacement implementation evidence

Active phase: verification; branch finishing blocked by existing full-suite failures. Bound scope: design commitment v1.

## Approval and source

User approved with "진헹하자" after the final design approval request. Approval binds
`docs/superpowers/specs/2026-09-16-evidence-replacement-design.md` at commit
5557bdd1ef41ea8e5354a2ac1990b636f0d5cf19, SHA256
8fa8e36fb433ebe2e7563852ee309ce1ab2e97fda6f6a30bc6841ab1983e1f4e.
The scope is implementation and engineering verification; no training launch.
The user accepted the stated experimental uncertainty in that approval request.

Implementation worktree: `/root/workspace/ILVR/.worktrees/ours1-evidence-replacement-20260916`.
Branch: `feat/ours1-evidence-replacement-20260916`; integration base is `ours1`.
Snapshot baseline: `ae5a1cd`; it includes the source's existing tracked changes
and seven relevant untracked files. Snapshot manifest and hashes:
`results/evidence-replacement-20260916/source-snapshot.json`. Original tracked diff
SHA256 remains f58f8469d938111814c4efe71b075126aa24bc7580b286289ee1344f37131ff7.
No new feature is attributed to the baseline-copy commit.

## Workflow and baseline

Design lifecycle/hook evidence is in the bound design document. Root executed
writing-plans and using-git-worktrees; worktree consent came from the approved
design, and no native worktree creation tool was exposed. Reused repository venv
and data/checkpoint symlinks. The SDD task/interface self-review table lives in
this plan's scoped ledger and will be preserved here before scratch cleanup.

Baseline command:

```bash
PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q -m 'not gpu and not distributed' tests/step_preference/test_config.py tests/step_preference/test_context_preference.py tests/step_preference/test_context_sidecars.py tests/step_preference/test_evidence_gain.py tests/stage2/test_launcher.py tests/multinode/test_evidence_gain_forwarding.py
```

Result: **204 passed, 2 deselected, 161.61s**, exit 0. Complete output:
`logs/evidence-replacement-20260916/baseline.txt`. This is a focused source
baseline, not a full-project or new-feature pass.

Implementation and task reviews are finished. Whole-feature review and one fix wave are recorded below. The project full-suite command failed; finishing does not proceed to an integration choice.

Plan shower: `/root/replacement_plan_cold`, context-free artifact copy, full read; minor repository-discovery gaps only. Root clarified fixture initialization, failure type, RNG device scope, skill location, baseline relationship and workflow-record location. No design behavior changed.

## Final implemented and reviewed revision

Feature/source baseline: ae5a1cd. Feature commits:6e49a5e,25536e9,de16010,606c854,3112e96. Verification-only fixture repair:4b27265907a595278855b5382a78d263171ecc84. No production file changed after3112e96. Root authored only workflow/control artifacts; implementation/fixes were delegated. Agent IDs, models, reviews, RED/GREEN commands and results are preserved in [records](2026-09-16-evidence-replacement-records/progress.md).

Task1 review fixed3validation findings; Task2 and3 approved with minors. Whole-feature review found ineffective donor-copy gradient regression; the single fix wave added detached literal-gradient comparisons and an in-memory mutation witness, mode-specific metadata gating, distinct target scopes and T-off rejection coverage. Scoped review left one automated independent-scope-score assertion incomplete.

Ruling: retain that automated-regression gap as a real minor, obtain independent runtime evidence instead of a second product fix wave, and do not claim the assertion exists. Eight score comparisons across both distinct scopes matched native literal replay exactly; exact command/code/output in [scope verification](2026-09-16-evidence-replacement-records/target-scope-verification.md). Risk if this disposition is wrong: a future scope-scoring regression might evade the current automated Trainer assertion.

## Final verification outcomes

- At3112e96, the planned full project command returned **6 failed,3297 passed,6 skipped,5 deselected,11 warnings**,849.83s,exit1. Full output: logs/evidence-replacement-20260916/final-cpu.log. Despite the gpu-marker exclusion,2 old CUDA tests lack that marker and ran. This was not a pure CPU-only run.
- One new fixture incompatibility was causally reproduced and fixed by restoring the actual constructor's default config; independent review passed. Production save behavior unchanged.
- Fresh focused verification on4b27265: **141 passed,1 skipped**,12.86s,exit0. Command: `PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q -m 'not gpu and not distributed' tests/step_preference/test_config.py tests/step_preference/test_replacement_plan.py tests/step_preference/test_replacement_runtime.py tests/step_preference/test_replacement_trainer.py tests/step_preference/test_stage2b_trainer.py tests/step_preference/test_shared_prefix_model.py`. Full output: final-focused-after-repair.log in the same log directory.
- Fresh new-mode CUDA on4b27265: **1 passed**,12.15s,exit0. Command: `RUN_REPLACEMENT_CUDA=1 PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_replacement_runtime.py::test_cuda_flex_dispatch_matches_dense_outputs_and_gradients`. Full output: final-cuda-after-repair.log.
- Three non-CUDA failures reproduce before this feature atae5a1cd. Both old CUDA cases pass isolated on baseline/current and fail under forcedTF32=True on both; that state sensitivity predates the feature. The original suite's precise TF32 state was not recorded, so the preceding un-restored setter is an inference. [Diagnosis](2026-09-16-evidence-replacement-records/baseline-diagnosis.md).

The full suite was not rerun merely to reproduce known baseline failures; no full-suite pass is claimed. Required finishing Step1 remains blocked. No merge, push, PR, training, efficacy, full-size feasibility or live distributed-metric claim. PEFT metadata save wiring was traced; a real PEFT save fixture remains unexecuted.

Source-preservation checks: original ours1 tracked diff SHA256 remains f58f8469d938111814c4efe71b075126aa24bc7580b286289ee1344f37131ff7; approved design SHA256 remains8fa8e36fb433ebe2e7563852ee309ce1ab2e97fda6f6a30bc6841ab1983e1f4e. Full logs/source manifest/probe hashes in [artifact manifest](2026-09-16-evidence-replacement-records/artifact-sha256.txt). SDD scratch is retained because the branch lifecycle is not cleanly finished; evidence is also tracked here.
