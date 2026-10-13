# SDD ledger — plan: docs/superpowers/plans/2026-10-13-dil-bce-orig.md

Approval: user `Subagent` approves the presented plan at e597354f09b9d4f2db4925cb48ff55e9157f8da4 and selects Subagent-driven. Plan SHA256 e699acb7432bb4d116ab1b9f64c7e911ddbe45026ed4593d3f27ea92b5310722. Written spec approved earlier at 112b09c. No training execution authorized.
Workspace: /root/workspace/ILVR/.worktrees/dil-bce-orig. Branch dil-bce-orig. No prior task records exist.
Skills executed: using-superpowers lifecycle reconciliation; using-git-worktrees existing isolation check; SDD setup/plan scan; requesting-code-review routing. TDD/test-quality read and required for implementers. Controller authors only control records; no product edits.

## Preflight task/interface scan

| Tasks | Producer versus consumer / shared file | Finding |
| --- | --- | --- |
| 1,2 | Numerical kernels vs config constraints | Both require orig fixed beta/margin and detached reference; no direct coupling |
| 1,3 | Clean/ref logits shapes | Replay produces P,V full vocabulary consumed by kernels; agrees |
| 1,4 | Frozen ref ownership | Loader output remains detached before numerical helper; agrees |
| 1,5 | OrigTokenComponents/CandidateSupport/entropy/mass and topk_entropy_loss.py | Task5 extends scalar-stat dataclass only; helper ownership stays Task1 |
| 1,6 | Probe same-forward numerical oracle | Probe uses literal expression independently; does not reuse kernel as expectation |
| 2,3 | Profile validation vs replay modalities | Unsupported profiles fail, native/PEFT supported; agrees |
| 2,4 | reference_identity binding vs ReferenceSnapshot | fingerprint+compatibility binds after snapshot verification; path separately stored |
| 2,5 | StepPreferenceConfig/identity reduction tag | reducer uses orig global microbatch step mean plus occupancy and fixed GAS; agrees |
| 2,6 | CLI fields/config/dil_identity | worker must retain explicit/omitted provenance; bind identity before Trainer save |
| 3,4 | CleanHistoryReplay -> FrozenCleanReference.score | same packet and replay helper; no separate rollout or student KV |
| 3,5 | model request/output; modeling_qwen2_5_vl.py | capture after student rollout; stats added in Task5; clean-first ordering retained |
| 3,6 | native/PEFT wrapper resolution | main passes separate owner, not model child; agrees |
| 4,5 | FrozenCleanReference -> Stage2Trainer | plain owner outside optimizer/DDP; agrees |
| 4,6 | snapshot/owner -> coordinator/main | one writer before workers; worker only validation; resume original provenance vs local paths |
| 5,6 | trainer.py reduction/metrics -> save/probe callbacks | Task6 observes raw steps without changing normal gradient path; numerical oracle separate |
| 1 | Internal tests/code/files | Step3 includes chunk API before Step4 chunk RED: reorder only this TDD substep as ruled below |
| 2 | Internal tests/code/files | CLI nullable fields and explicit null need explicit_fields; plan names it; matches identity tests |
| 3 | Internal tests/code/files | owned packet clones preserve live student graph; literal ref oracle independent; consistent |
| 4 | Internal tests/code/files | immutable copies versus relocated effective base: retain provenance bytes, resolve load path separately; consistent |
| 5 | Internal tests/code/files | Step4 raw reduction is covered by trainer RED; distributed counts added after Gloo RED in Step7; consistent |
| 6 | Internal tests/code/files | worker/probe RED precedes implementation; prose documentation does not earn tests; runtime launch excluded |

Ruling: Task1 writes chunk/checkpoint assertions before implementing orig_components_chunked (move that portion of Step4 into initial RED, or implement the chunk wrapper only after Step5 RED) — the TDD skill prohibits code preceding its failing witness; equations/interfaces and scope are unchanged — cost if wrong: only test sequencing/rework, no objective change.

## Task state

- [x] Task 1: numerical kernels — independent review complete
- [ ] Task 2: config/identity
- [ ] Task 3: replay
- [ ] Task 4: reference artifacts/owner
- [ ] Task 5: Trainer/reduction
- [ ] Task 6: entrypoints/probe/docs
- [ ] Final independent review and fresh verification

Resource witness: nvidia-smi showed local H100 NVL GPU0/GPU1 at 0 MiB and 0% with no compute processes. Each full-suite runner rechecks before CUDA unit tests; no training or remote execution authorized. Existing canonical B200 wording is historical environment description, not permission to change packages.

Task 1: dispatched /root/orig_task1_impl, model gpt-6.1-sol/high; BASE c75d9ed58d7dc343e9ceec3117517696e9ba4b46. Brief task-1-brief.md; report task-1-report.md. No earlier implementation reused.

Task 1: observed focused GREEN 68 passed in 1.71s; confirmed missing-kernel RED 45 failed in 1.33s. Logs logs/dil-bce-orig-t1-20261013t163441z/{confirmed-red,green}.log. Full suite had begun but has no completed verdict.
Ruling: fix the self-review-discovered saturated entropy NaN using a causal RED and valid q>0 masking before multiplication; interrupt only the implementer's owned suite and rerun after the fix — spec N2 requires stable saturated entropy and the existing helper supplies the proven pattern; no floors/clamping or arbitrary precision redesign — cost if wrong: an extra focused/full test run and possible further numerical boundary refinement.

Task 1: implementer completed project-wide CPU invocation with CUDA_VISIBLE_DEVICES empty, contrary to the full-suite GPU requirement. Observed exit1: 4035 passed, 38 skipped, 15 failed, 33 errors in1224.79s; all 48 fail/error node IDs and traces preserved in logs/dil-bce-orig-t1-20261013t163441z/final-full-suite.log. Trace causes were FA2 unavailable on CPU, packed CUDA assertions, and no CUDA GPU. Root read the complete short failure/error summary. This run is not a green full-suite witness; GPU-skipped coverage is not claimed.
Task 1: minimal environment discrimination on unchanged code: exact48 failing/error cases with CUDA_VISIBLE_DEVICES=0 all passed in12.28s, exit0/no warnings. Command/environment JSON and log: gpu-rerun-command.json and gpu-rerun.log under that same log directory. Future full-suite invocations explicitly expose a checked idle GPU. No unrelated source fix was made. This is a test-environment execution correction, not authorization to launch training.

Task 1: implementer DONE_WITH_CONCERNS at 5debf2ad7e129ca4e2a5264a0217064cf6546761, four assigned files/557 lines; all50 new tests passed in the project-wide CPU invocation, final focused73 passed in1.72s. Concern: no single green full-suite invocation due CUDA-hidden execution, exact48 failure/error cases separately resolved on GPU0; GPU-skipped coverage not claimed. Report task-1-report.md and logs retained.
Task 1: reviewer /root/orig_task1_review, model gpt-6.1-sol/high, dispatched with task brief, verbatim constraints, report and review-c75d9ed..5debf2a.diff. Spec/quality verdict pending. Root did not inspect/review the product diff or substitute for independent review.

# Task 1 independent review
Reviewer: /root/orig_task1_review; model gpt-6.1-sol/high.
Reviewed BASE c75d9ed58d7dc343e9ceec3117517696e9ba4b46 to HEAD 5debf2ad7e129ca4e2a5264a0217064cf6546761.
Spec compliance: issues found. Task quality: needs fixes.
Important finding (verbatim):
`src/step_preference/dil_bce_orig_loss.py:42` — vocabulary-bound comparison wraps for narrow integer targets. The validator explicitly accepts uint8/int8/int16, but PyTorch converts the Python vocabulary-size scalar to the tensor dtype during comparison. A focused runtime probe rejected valid target `0` for uint8/V256, int8/V128, and int16/V65536; int32/int64 controls passed. Both BCE and `build_entropy_support` inherit this defect. Widen targets to long before the bounds comparison and add a causal regression covering those dtype/vocabulary boundaries.
No Critical or Minor findings.
Strengths: full-reference BCE/detach; token-level chunk concatenation and checkpointing; detached GT-union candidates; live stable log_softmax entropy; stable mass reuse with live complement gradient.
Cannot verify: new-kernel GPU/production shape memory/cost are later acceptance gates; full CPU invocation remains red, separate48-case GPU rerun is not a green full-suite claim.
Named review checks: one focused dtype-boundary runtime probe; unchanged _group_mass_terms dependency inspection. Common-offset entropy risk resolved by actual stable log_softmax path, no extra probe. No suite reruns or code mutations.
Controller resolution of cannot-verify: those production witnesses are assigned Tasks5/6 and final acceptance, not proof supplied by Task1. Do not claim them passed; final full-suite command must use a verified idle GPU. No spec failure is being waived; the Important dtype finding enters fix round1.

Task 1: fix round 1/5 started; FIX_BASE 5debf2ad7e129ca4e2a5264a0217064cf6546761; resume original implementer with the verbatim Important finding and focused regression tests.

Task 1: fix round 1/5 (1 addressed, 0 open; commits 5debf2a..c52fc56). Reviewer /root/orig_task1_review returned ADDRESSED for narrow integral target bounds, no new breakage/out-of-scope observations; inspected fix diff and saved RED12 failed/8 controls passed and GREEN93 passed, no rerun. Spec compliance and quality issues are resolved for this task's four-file deliverable. Production/new-kernel GPU and single green full-suite claims remain outside this task's evidence.
Task 1: complete (commits c75d9ed..c52fc56, review clean).

## Subsequent user scope and stage correction

User requested: `학습과 학습 후 자동 평가 작업 범위까지 진행해.` This explicitly authorizes training and automatic post-training evaluation as subsequent work after required development/review/verification/preflight. It does not pick reference weights or override coefficient/topology defaults.
Root initially proposed a reproduced Stage1 checkpoint as a Stage2 initializer/reference. User responded `Stage 1 학습인데 무슨 Stage 1모델을 초기모델로 사용한다는거지?`. Root stopped affected Stage2 progression and asked a stage-scope clarification. User explicitly selected `Stage 1 학습에 DIL-BCE-ORIG 적용`.

The latest instruction supersedes the previous Stage2 integration scope. The source document mentions Stage2B in section1.1; the previously approved spec/plan accurately represented that interpretation, but are not authority to implement Stage2 after this correction. No Stage2 integration task was dispatched: only the stage-independent numerical kernels and their tests were implemented/reviewed. Preserve those exact changes/evidence; do not reimplement Task1.

Active phase is now architectural brainstorming for a Stage1 revision, not execution of old Tasks2-6. The selected Subagent-driven implementation method persists. Read-only investigation and controller control artifacts may continue; Stage1 product integration awaits the revised applicable design/plan gates. No further training permission is needed merely to repeat authorization already given; unresolved protocol/design/reference choices still need resolution before execution. No training/evaluation has launched.

Canonical training-defaults.md was freshly read after execution scope expanded. Stage1 starts fresh from original Qwen2.5-VL-3B under its own teacher/EMA/ZeRO semantics, subject to the revised approved integration. Stage2 initializer checkpoint431 and Stage2 212-step budget must not be carried over as Stage1 defaults. Stage1 update budget must be measured under its actual retained data/sampler/Trainer behavior. Automatic evaluation retains canonical TEST400 defaults unless explicitly changed.
