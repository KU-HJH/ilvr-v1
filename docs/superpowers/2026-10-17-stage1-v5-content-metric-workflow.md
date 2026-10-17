# Stage1-v5 ILVR Teacher input and external content metric

## Intake and authorization

Understood as: replace current adaptive target selection so the target Teacher
uses ILVR/legacy placeholder-conditioned language observation; compute history
relevance externally from observed current-text hidden states and previously
selected raw visual targets. Preserve frozen DoRA/EMA1 weights, reference role
and execution rules, all losses and the legacy selector. No training requested.

User clarified `Target 선택용 teacher만 (권장)` and corrected terminology:
Teacher means target selection; reference means DIL scoring. Root accepted this.
After the architecture, cosine formula, alternatives and verification direction
were presented, the user requested `그렇게 현재 구현을 수정하자.` This confirms
the preliminary direction, not final approval of a subsequently written design.

Current workspace: `/root/.codex/worktrees/6b9d/ILVR`, existing isolated named
branch `Stage1-v5`, intake `26ece5c`. Tracked files clean; existing untracked
caches and cancelled retrieval workflow are not owned by this task. Canonical
AGENTS read. Existing venv/data/checkpoint links preserved. Prior merged-source
full suite:4935passed/11skipped/1warning at product merge `1c86472`; no fresh
baseline or implementation has run in this design phase.

Active phase: Superpowers brainstorming, written-spec path. Relevant skills
read: using-superpowers, readchk (scope reconciled), brainstorming, hate and
macrothink. Root alone owns advisory hooks. Retain the user's selected
Subagent-driven implementation method after final approval; do not switch it.

## Lifecycle checklist and evidence

1. Context: actual target-builder legacy/adaptive calls, shared frozen model
   roles, prefix injection, metric aggregation, parser/launcher and strict
   query/loss resume contracts inspected. Public pinned ILVR source and paper
   were checked in the preceding assessment. The claim is matching current
   ILVR/legacy input/forward semantics at fixed weights, not original training.
2. Visual companion: nonapplicable; no UI/visual artifact is requested.
3. Preliminary scope: Teacher only; reference unchanged; frozen weights kept.
   Existing recommendation's architecture/dataflow/math/verification direction
   was accepted by the user's implementation request. Routine empty-input,
   configuration and resume details are explicit in the written proposal and
   remain subject to the final written commitment approval.
4. Alternatives: external parameter-free cosine attention recommended;
   placeholder-position native attention does not measure selected visual
   content; reusing Q/K projections adds representation assumptions. User
   accepted the recommended direction. Performance superiority is unproven.
5. Complete proposal and self-review: draft below; freeze before critique.
6. Root hate; conditional macrothink: pending. Wrong semantics could invalidate
   future multi-hour training, so three bounded fresh reads are justified.
7. Synthesis into exact design commitment, final human approval: pending.
8. Writing plan, SDD/TDD, independent task/final reviews: not started.
9. Fresh completion verification and applicable finish: not started.

Control artifacts may be authored by root. No requested implementation,
test, CLI or deliverable documentation change is authorized before final
commitment approval. No premise probe, dependency install or training runs.

## Frozen proposal and advisory gate

Complete proposal v1 is
`specs/2026-10-17-stage1-v5-content-metric-proposal-v1.md`, SHA256
`7748b23f4973d235fb827b8419abc35ff24115c21dab37fa0ef882633f53c6d8`.
Root self-reviewed scope, role/input distinction, metric/mixture equations,
zero/empty cases, saved identity migration, tests and unknowns. Proposal bytes
are frozen before critique; all passes read this same version.

Root executed hate and captured exactly `{root, first_nail}` in the adjacent
JSON. Root objection: cross-representation cosine may not carry calibrated
relevance, while unit-normalized history would always enter the query. The
proposed first nail needs real model/sample observations and potentially
paired next-text loss. No such probe is authorized or run by this design task;
no PASS/FAIL or semantic evidence is inferred from the proposal.

Macrothink condition is satisfied: solving the wrong relevance objective
could invalidate future multi-hour training. Three bounded read-only fresh
sessions cost less. Root strips prior examples/naming from the initial goal
prompt, asks independent readings, then has each inspect this identical frozen
proposal. No model/probe execution, edits, hook invocation or sub-agents by
fresh readers. Results and classifications will return to brainstorming;
they cannot authorize implementation or substitute for final human approval.

## Critiques synthesized; final written approval gate

Fresh readers `/root/content_metric_fresh_a`, `/root/content_metric_fresh_b`,
`/root/content_metric_fresh_c` all completed read-only. Each independently
confirmed the frozen proposal hash and returned no demonstrated incompatible
defect. Root classified their compatible divergences as the same calibration/
informativeness root, with selector-boundary, vector-norm, row-prior, call-scope
and strict historical-entrypoint refinements. Convergence is reassurance only.
Detailed findings/classification/limits are retained in the critique record.
No reader ran a model, test or probe, edited source or invoked another hook.

Root synthesized commitment v1 at
`specs/2026-10-17-stage1-v5-content-metric-design.md`, exact SHA256
`113658fdd4c8ddf8357d3cace7dff21753e8148969a4c566961357a05aaa1d3d`.
Relative to the frozen proposal it adds an actual selected-index/count witness,
clarifies language-wrapper count and equal-coefficient/relative-relevance terms,
and separates old native-mask identity from new placeholder/native-mask identity.
Final self-review clarified that a semantic control must mismatch text or
weight/value association, not merely permute history rows jointly with recomputed
scores. Original hate evidence is preserved. Proposal bytes remain unchanged.

Self-review checked exact formulas, gamma/tau, empty/nonfinite behavior, role
boundaries, current/reference input rules, source/resume identity, CLI migration,
independent tests, all critique dispositions and unsupported efficacy claims.
No placeholders. First nail is NOT RUN; relevance/accuracy and production
runtime are unverified. Exact commitment requests uncertainty acceptance for
implementation/testing only, not real training or a costly premise probe.

Root is committing only these control artifacts for written-spec review.
No requested code/test/CLI implementation has changed and writing-plans has
not started. Final human approval of this exact commitment is pending.
Applicable mandatory skill gate: brainstorming/SKILL.md says to ask the user
to review the written spec, then "Wait for the user's response" and "Only
proceed once the user approves." The earlier implementation request accepts
the direction but does not substitute for this later exact written commitment.

## Written commitment response and planning transition

After the exact written commitment and its implementation-only uncertainty
were presented, the user asked the next phase and then explicitly invoked
`$superpowers:writing-plans`. Root treats that instruction to advance the
presented design as approval to plan the unchanged commitment v1 above. This
records the actual response, not a claim that the user selected a suggested
approval-button phrase. The frozen spec retains its creation-time pending
status; this later entry is the authoritative phase transition. No evidence
about metric usefulness, production cost, or accuracy has been added, and no
training, evaluation or real-data premise probe is authorized.

Active phase: writing-plans; implementation method remains the previously
selected Subagent-driven method. Skills writing-plans and using-git-worktrees
were read and applied. Existing isolation verified: named `Stage1-v5`, Git dir
`/root/workspace/ILVR/.git/worktrees/ILVR1`, common dir
`/root/workspace/ILVR/.git`, no superproject. No new worktree/dependency install.
Planning starts from control commit `b39e0f2`, product merge `1c86472`.

Fresh focused baseline command (repository venv, vendored Transformers first):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/test_history_query.py tests/stage1/test_history_targets.py tests/stage1/v5/test_resume.py tests/stage1/v5/test_history_integration.py
```

Observed exit 0, `176 passed in 11.87s`; complete output read from
`logs/stage1-v5-content-metric-20261017/baseline.log`. This is the pre-change
baseline, not evidence for the proposed behavior. No product file changed.

Read-only consumer tracing found the verification script's source-hash map
already lags the runtime map (three trainer keys versus seven runtime keys).
The implementation plan must align the actual map, including the new metric,
and retain exact hash checking; synthetic three-key fixtures are insufficient.

## Implementation plan and root self-review

Plan saved at `plans/2026-10-17-stage1-v5-content-metric.md`. Two sequential
tasks: pure unused metric with independent math/selector tests, then atomic
adaptive activation plus strict identity/resume/consumer updates. The latter
cannot be separated into a misleading new-formula/old-runtime intermediate.
The plan includes exact paths, APIs/code snippets, RED/GREEN commands, reference
and DIL invariants, independent task/final review and final suite/handoff gates.

Root self-review matched every commitment section to task/verification evidence,
checked API/type consistency and corrected fixed-temperature JSON type checking
and actual hidden-state hook capture. Fresh mechanical plan check: two tasks,
balanced fences, no placeholder patterns, complete metric module AST parses;
exit0. `git diff --check` exit0. This validates the plan artifact only, not the
proposed implementation. No product files have changed at this point.

The previously selected Subagent-driven method is retained without another
method-selection question. Next phase is SDD/TDD with fresh implementer agents,
spec-before-quality task reviews and an independent final whole-change review.

## SDD Task 1 in progress

Root read/applied SDD, TDD, requesting-code-review and reviewer templates.
Fresh implementer `/root/content_metric_math_impl` received the complete task
requirements in a fresh-context dispatch; only two new pure metric/test files
are owned. Base `9263b33b19913dd11dca6f39ffaaa9320937b25f`. No product activation.
Implementer reported the literal missing-module RED and first arithmetic GREEN.
It then added a CPU enclosing-autocast regression: FP32-cast inputs still ran
BF16 matrix multiplication and failed expected arithmetic tolerance (~3.4e-4).
Root authorized the local autocast-disable correction as necessary to meet the
already-approved FP32 formula; no new scope or semantic calibration evidence.
Plan clarification added. Full evidence/review verdicts remain pending.

Root located the real historical adaptive sidecar read-only at
`checkpoints/stage1-v5-adaptive-ce05-al05-g09-last-e3-w4-20261016-073334/checkpoint-648/stage1_v5.json`.
It carries outer DIL schema1 and query schema1/native injected-prefix. Task 2
will preserve and inspect it; a separate outer schema2 fixture is needed to
isolate the native-query training rejection from the existing block-mean guard.

Task 1 implementer completed at `9d1b03aa9394c18378fefe1c07b5745a0dbdfd5c`;
root inspected actual two-file diff (97 module lines, 223 test lines) and read
complete RED/GREEN logs: missing module, empty/validation/composition failures,
caller-autocast arithmetic mismatch, malformed present-history summary failure,
then final `167 passed in 3.30s` (exit0). Runtime remains unactivated. Scope
contains only `history_metric.py` and its tests. Selector witness reports metric
index0 versus uniform index1 with positive margins; no semantic usefulness claim.

Fresh independent specification reviewer `/root/content_metric_math_spec`
dispatched read-only against base `9263b33` and exact product `9d1b03a`.
Review verdict pending; quality review and Task 2 cannot advance before PASS.

Task 1 review gates completed on exact product `9d1b03a`:
`/root/content_metric_math_spec` returned SPEC PASS after independent source/
scope/log review and fresh `167 passed in 3.31s`, exit0. Then distinct
`/root/content_metric_math_quality` returned QUALITY PASS with no findings;
its targeted CPU BF16 dispatch probe observed FP32 score/mean/norm operations,
exact no-autocast output equality and no gradient path. Neither reviewer edited
files or ran training. No unresolved findings. Task 1 checked steps refer to
these actual witnesses, not implementer assertion alone.

Active phase advances to SDD Task 2: atomic adaptive activation, saved identity,
historical read/current training split and affected consumer/integration/docs.
Task 1 arithmetic review is reused; Task 2 and final review remain required.

Fresh Task 2 implementer `/root/content_metric_activation_impl` started from
`eea7a3c043d963a766847806cc6f5654248de9c4` with complete fresh-context task text.
Observed causal RED logs: legacy literal witness catches three decoder prefix
calls and zero Teacher wrapper calls; current schema/layer RED catches schema1
and last4/all acceptance. Initial query/literal observation/frozen legacy GREEN:
113 passed. Later target semantics GREEN:39 passed (task still in progress).

Actual source-file regression caught `inspect.getsourcefile` of the no-grad
decorated metric resolving to the PyTorch wrapper. Implementer is binding the
actual metric source through its class/module; retain this causal witness.
Implementer also found affected hardcoded source/schema assertions in
`tests/stage1/v5/test_trainer.py`, absent from the initial inventory, and asked
before editing. Root authorized only the affected consumer assertions/fixtures,
updated plan file map and focused command. This is within existing identity/
consumer scope, not a changed design or self-authorized scope expansion.

Task 2 implementer returned DONE_WITH_CONCERNS at
`cc4336f7a5752152f83a0083e2d9c4468649561e`. Concerns are the already-disclosed
absence of real-data relevance/calibration, cost and accuracy measurements;
no additional unresolved correctness finding was reported. Root inspected the
actual 17-file diff (736 insertions/361 deletions), including Teacher observation,
query reconstruction/training dispatch, source maps, README and parent/reference
tests. No loss computation or data/sampler/worker code changed.

Root read complete `task2-focused.log`:548 passed in18.85s, exit0, including
the authorized trainer consumer. Causal observation/schema RED logs previously
read; remaining causal/source/integration logs retained alongside. Real sidecar
post-change strict read PASS with unchanged SHA256
`8e064ae861790494e9f3142e49f39a95756eae3e579bb9792d2941beccfcf4f9`;
root read complete inspection output and earlier baseline metadata is under
`results/stage1-v5-content-metric-20261017/historical-before.json`.

Fresh independent `/root/content_metric_activation_spec` dispatched against
exact `eea7a3c..cc4336f` with bound requirements and entrypoint/evidence discipline.
Specification verdict pending; quality and final review not yet authorized to
advance. No whole-project suite or training has run in Task 2.

Task 2 review gates completed on exact product `cc4336f`:
`/root/content_metric_activation_spec` returned SPEC PASS after inspecting all
17 files, tracing Teacher/helper/history and parent/reference/loss paths, and
independently checking the real historical sidecar. It reports28 targeted
witnesses passed in10.84s, exit0; final full-suite verification remains required.
Then distinct `/root/content_metric_activation_quality` returned QUALITY PASS,
no critical/important/minor findings, after reading every changed-file diff and
tracing actual calls/state/guards/source-map consumers. Fresh diff check PASS;
it reused the complete548-test focused log without another broad run. No edits
by either reviewer. Task 2 completed with no unresolved review finding.

Active phase: required final independent whole-change review, covering both
metric `9d1b03a` and activation `cc4336f`, before final full-suite verification.
Unmeasured real-data relevance, GPU behavior/cost and accuracy remain unknown.
No training/evaluation/real-data premise probe has been launched or authorized.

## Final whole-change review and verification selection

Distinct fresh `/root/content_metric_final_review` returned final whole-change
PASS on `b39e0f2e4cea04dd8241a8ae358c4ca4ff0ee062..38d3852edde07876bde900224859195352afa9d2`.
It inspected actual code/call/data/state and changed tests, read the complete
548-pass log, checked the historical sidecar hash and diff hygiene; no edits or
test-suite repetition. Critical/Important/Minor: none. Product is still exactly
`9d1b03a` plus `cc4336f`; all independent gates are green, no unresolved finding.

Root applied verification-before-completion and finishing-a-development-branch;
announced both completion gates. Finishing applies to the feature handoff and
its Step1 project full suite supplies the fresh test-pass witness for both
gates. No extra full suite is added. Workspace is named/host-managed, Git dir
differs from common dir; `main` exists with merge-base
`2497ee7bafad1b51ef6c2aba6f40f084830916c4`. Human retains branch disposition;
no worktree deletion/merge/push is authorized by this verification.

Selected command, after final review and before any completion claim:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q --basetemp=checkpoints/stage1-v5-content-metric-20261017/final-pytest-r1 > logs/stage1-v5-content-metric-20261017/final-full-suite.log 2>&1
```

Pre-command guards freshly PASS: checkpoint link preserved; task-owned basetemp
resolves beneath canonical checkpoint tree and does not exist (pytest clears
explicit basetemp); output log is new; exact product-file equality against
reviewed `38d3852` and `git diff --check` both exit0. Plan command updated only
to keep any temporary scalar/model fixtures under the checkpoint namespace.
This is not a product or formula change. Frozen proposal/commitment hashes
rechecked unchanged. Full-suite result remains pending; no completion claim.

## Verification attempt 1 environment failure and targeted recovery

At control revision `9ab1c5d`, full-suite session8401 failed fixture setup because
root added a nested explicit basetemp but did not create its parent. Root
observed widespread E markers and stopped only its own verified pytest process
(PID2537855, parent2537853, exact argv rechecked before SIGINT). Exit2 after
228 passed/183 fixture errors in85.04s. No foreign process was signaled.
Failed output remains `final-full-suite.log`; root read the entire141549-byte,
2757-line log programmatically, extracted every exception, and confirmed all183
are the identical missing-parent FileNotFoundError. This is not a product
regression or a passed full suite; no implementation/review gate was replaced.

Smallest fix: create only verified task-owned checkpoint parent. No code edits.
Causal preflight uses the first failed fixture, same pytest basetemp mechanism:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/evaluation/test_continuous_latent.py::test_external_evidence_loader_does_not_write_copied_source --basetemp=checkpoints/stage1-v5-content-metric-20261017/pytest-path-preflight-r2
```

Observed exit0,1passed in2.73s; complete output read at
`pytest-path-preflight-r2.log`. Retry uses fresh basetemp `final-pytest-r2` and
`final-full-suite-r2.log` with the same full-project command/profile. Fresh
preflight confirms parent exists, canonical path, and unused log/base directory.
Plan preflight corrected. Identical reviewed product/source remains unchanged;
all task/final review verdicts still apply. Retry is required by this concrete
failed verification, not an optional duplicate broad cycle. Result pending.

## Full-suite attempt 2: one stale launcher test consumer

At exact control `c2ca892`, session92703 completed exit1:
`1 failed,5077 passed,11 skipped,1 warning in1218.88s (20m18s)`.
Root read the complete log `final-full-suite-r2.log`. Sole failure:
`tests/stage1/v5/test_launcher.py::test_actual_coordinator_adaptive_resume_omits_query_defaults_on_both_nodes`.
Its imported `adaptive_contract()` fixture now correctly represents schema2/last,
actual coordinator and worker parser succeed, but line276 still expects last4.
The consumer was outside the initial focused file inventory. This is a real
failed completion witness, not a completed suite; final integration gate stays
closed. Warning is the existing CPUAdam TORCH_CUDA_ARCH_LIST message.

Root's live status checks read only the tail, so earlier progress updates missed
the F marker printed at45%. Root corrected that overbroad no-failure status on
reading the complete result. Subsequent progress checks must inspect all existing
progress output for F/E rather than infer whole-run status from the tail.

Reopened Task2 narrowly: root explicitly authorized original implementer
`/root/content_metric_activation_impl` to migrate only affected assertions in
`tests/stage1/v5/test_launcher.py`, preserving both-node coordinator/default
omission/gamma/parser/resume witnesses. This is another current-contract
consumer inventory correction, not a changed design. Full-suite failure is its
causal RED. No production file change is requested. Original independent task
and final reviewers will review this delta; their identical-product assessments
are reused. Plan map/focused set corrected; fresh whole-suite retry required
after the fix/review gates, with a new log/basetemp identity. No training.

## Launcher delta fixed and independently re-reviewed

Original implementer committed only the authorized consumer file at
`2b12b43cc3ef6d34d78e158df4455fd9b4f821d8`:four insertions/one deletion.
Fixture explicitly binds queryschema2/exactformula; last4 assertion becomes last.
All actual both-node/four-rank dryrun, omitted defaults, saved gamma and paths
remain intact. Root inspected the full diff and complete RED/GREEN logs:
`task2-launcher-consumer-red.log` reproduces1failed in3.22s at the stale assertion;
`task2-launcher-consumer-green.log` runs the entire launcher file,9passed in3.50s,
exit0. No production file changed.

Original independent reviewers assessed only this new delta while reusing
identical reviewed code: `/root/content_metric_activation_spec` SPEC PASS,
then `/root/content_metric_activation_quality` QUALITY PASS, then
`/root/content_metric_final_review` extended whole-change PASS at exact2b12b43.
Each inspected actual changes/evidence; no outstanding finding or reviewer edit.
Historical migration and original production approvals remain applicable. This
is required fix/review, not a replacement or waiver of any earlier gate.

Active phase returns to verification-before-completion and finishing Step1.
Fresh command at the reviewed code/test revision (control-only record follows):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q --basetemp=checkpoints/stage1-v5-content-metric-20261017/final-pytest-r3 > logs/stage1-v5-content-metric-20261017/final-full-suite-r3.log 2>&1
```

Fresh preflight PASS: parent exists; checkpoint link preserved and canonical;
basetemp/log unused; product and tests match reviewed2b12b43; diff check clean.
All previous failure logs preserved. No completion or merge claim before the
complete result. The required retry is justified by the failed completion
witness and actual reviewed test fix. No training/real-data probe authorized.

## Fresh completion verification PASS and human handoff

Full r3 command above ran at control revision
`5d33b2dc9f701f5561ebb246f5edb94bd93e975c`, product/tests identical to final-reviewed
`2b12b43cc3ef6d34d78e158df4455fd9b4f821d8`. Session64723 exited0. Root read the
complete output: **5078 passed,11 skipped,1 warning in1220.29s (20m20s)**.
The warning is the existing CPUAdam TORCH_CUDA_ARCH_LIST warning; skips are
reported as skips, not executed passes. Log `final-full-suite-r3.log`, SHA256
`e17f7a95a1eec5b18cce33e98bd6cf202433069da4dbbca492c4570ae0824377`.
Earlier attempts remain recorded failures; they are not completion evidence.

This one successful full-project witness after required fix/reviews supplies
both verification-before-completion's test-pass claim and finishing Step1.
It also runs all claim-specific metric/literal Teacher/selector/causality/
parent/reference/DIL/resume/hash/legacy/throughput witnesses at the same revision.
No duplicate suite is needed for overlapping claims. CPU contracts establish
mechanics only; no production GPU equivalence, relevance, accuracy or cost claim.

Fresh post-suite checks all PASS:

```bash
git diff --exit-code 2b12b43cc3ef6d34d78e158df4455fd9b4f821d8 -- src tests transformers/src scripts README.md run_stage1_training.sh run_multinode_stage1_training.sh
git diff --check
sha256sum logs/stage1-v5-content-metric-20261017/final-full-suite-r3.log docs/superpowers/specs/2026-10-17-stage1-v5-content-metric-design.md docs/superpowers/specs/2026-10-17-stage1-v5-content-metric-proposal-v1.md checkpoints/stage1-v5-adaptive-ce05-al05-g09-last-e3-w4-20261016-073334/checkpoint-648/stage1_v5.json
```

Frozen commitment/proposal and real historical sidecar hashes exactly match
the recorded values. Tracked working tree was clean before this control-only
completion record; original untracked caches/cancelled retrieval record remain.
Workspace Gitdir/common-dir still identify the named host-owned worktree; keep
it intact. `main` base candidate was detected, but human owns integration choice.

Implementation complete with independent spec/quality/final reviews and fresh
verification. Active phase: finishing Steps2-4, awaiting human branch disposition
(local merge, push/PR, keep, discard). No merge, push, cleanup, training,
evaluation or real-data premise probe has run. Planning approval binds to exact
commitment v1; semantic usefulness uncertainty remains as disclosed.
