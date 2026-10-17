# Stage1-v5 DIL token-mean workflow

## Intake and active phase

Understood as: change the DIL-BCE reduction in the existing `stage1_v5`
worktree from equal original-block weighting to equal eligible-token weighting.
Keep tokenwise softplus before averaging. The conversation concerns DIL;
changes to CE, alignment, entropy, mass, Text-off KL, query selection, training
coefficients, training launches, and existing checkpoint contents are outside
the requested change.

Source worktree: `/root/.codex/worktrees/1061/ILVR`, branch `stage1_v5`.
Inspected revision: `c9d6afe1293b404243e7a48d0e31aeaf33cab4e2`.
Tracked implementation was clean at intake. Canonical `AGENTS.md`, the user
supplied global workflow policy, and relevant skills were read.

Active lifecycle: approved commitment v1 implemented through the retained
Subagent-driven method; specification, quality and final reviews passed;
fresh full-project verification passed. Finishing is at the human branch
handoff choice, with no integration or cleanup authorized. Earlier pending
entries below describe their time; later evidence sections give current status.

The preliminary question distinguished a token denominator across all ranks
and all microbatches from averaging sample-local token means. The user answered
`전역 token 평균으로 진행 (권장)`. Follow-up questions about global/sample
weighting were answered using token counts; no reversal was requested. Other
terms keep their denominators. This preliminary confirmation is not final
written-spec/design-commitment approval.

## Context evidence

`v5_trainer.compute_loss` calls `compute_v5_loss_sums`. The orig kernel applies
softplus separately at each target token and reference-vocabulary entry;
`v5_objective` then sums per-block token means. `get_batch_samples` reduces
the original-block count across the prefetched accumulation window and four
ranks. `v5_auxiliary_scale` compensates for configured GAS and rank averaging.
DIL, entropy and mass currently share this block denominator; they must not be
silently changed together when changing only DIL.

`stage1_v5.json` schema 1 binds reduction to
`global_actual_window_original_blocks_empty_included_v1`. It strictly checks
all config fields and reconstructs the entire contract on load. A new default
cannot silently reinterpret historical resumes. Existing selection modes
include `all`, `clean`, `voff`, `gap`, and `intersection`; unlike `all`, selected
counts can depend on forward results and cannot simply be read from metadata.
This constraint must be resolved explicitly in the complete design.

Read-only reweighting of existing GPU measurements (no new forward/training):
`results/stage1-v5-dil-probe-20261017/{legacy,adaptive}.json` contains 37
nonempty original blocks and 1,753 DIL target tokens in the same 16 examples.
Weighting each measured block mean by its token count gives:

| Query mode | Original block mean | Whole-subset token mean |
| --- | ---: | ---: |
| legacy | 0.827538855978 | 0.984719363064 |
| adaptive | 0.827918072005 | 0.985221240514 |

These are numerical reweightings of previously measured per-block values,
not measurements of a new implementation or new training results. The
rounding of stored FP32 block means limits their precision.

## Lifecycle tasks and evidence status

1. Context exploration: complete; actual objective, window normalization,
   resume contract and existing selectors inspected.
2. Visual companion: nonapplicable; mathematical reduction/API behavior.
3. Preliminary scope/architecture confirmation: received global-token selection;
   later explanatory questions answered without a reversal.
4. Complete proposal and written commitment self-review: complete; exact artifacts
   below, final human written-spec approval pending.
5. Frozen proposal, root-only hate, conditional three-reader macrothink and
   critique synthesis: complete, with raw objection/status and classifications
   recorded in the adjacent artifacts. No implementation/probe result implied.
6. Final human approval of exact commitment: pending.
7. Writing-plans and retained user-selected SDD implementation, TDD,
   independent specification/quality review: pending final design approval.
8. Fresh claim-specific verification and applicable branch finish: pending
   reviewed implementation. No implementation, tests, training or completion
   is claimed.

Skills actually executed for intake: `using-superpowers`, `readchk`, and
active `brainstorming`. `hate` and `macrothink` were read for the mandated
later design gate but have not been executed for this change.

## Frozen proposal and advisory gate

Preliminary scope reply: `전역 token 평균으로 진행 (권장)`. Explanations
of sample versus token weighting were supplied; this is not final commitment
approval. The complete proposal is frozen at SHA256 `567d071472baa991f0b4a7f72616477b54fea4750026d36012afbe4bb39a41dd`.

Root executed hate against those bytes and captured exactly `{root, first_nail}`
in the adjacent hate JSON. The proposed CPU first nail has not been executed.
Macrothink condition applies: an incorrect accumulation reduction or silently
changed resume could invalidate a later multi-hour comparative training run.
Three bounded same-model fresh reads cost substantially less. Root stripped
the framing to user goal, runtime constraints and observed code facts; identical
independent read-only prompts dispatched to `/root/token_mean_fresh_a`,
`/root/token_mean_fresh_b`, `/root/token_mean_fresh_c`. They were instructed
not to edit/train/invoke advisory hooks/spawn. Results pending.

## Written commitment ready for final review

All three fresh-reader turns completed. Root classified incompatible conditional
intersection warnings separately from compatible fixed-cardinality corrections;
convergence is reassurance only. See the critique synthesis for agent findings
and the independently traced strict source-hash resume guard.

Final commitment v1 SHA256: `64d8a8474575e540f222514d9109836a2f2353ad5671fb4e672f0cea4075406e`.
Self-review inspected exact formula, two denominators, partial tails, empty
windows, selector scope, historical read versus early training-resume rejection,
new contract identity, independent tests and remaining uncertainty. No placeholders.

The final commitment simplifies the proposal by rejecting historical training
resume under new source while preserving historical read-only contract validation;
no new loss-mode CLI or generalized legacy engine is proposed. User final
commitment approval is still pending. No source implementation/test/training work
has been performed. Next gate: user written-spec review, then writing-plans
and the retained Subagent-driven route.

## Final approval and implementation plan

The user answered `진행해` after the exact final written commitment v1 was
presented with implementation approval requested. Approval binds SHA256
`64d8a8474575e540f222514d9109836a2f2353ad5671fb4e672f0cea4075406e`.
The frozen spec is not rewritten to erase its creation-time pending status.
This authorizes the described code/tests via the retained SDD method, not
training, evaluation campaigns, checkpoint mutation or integration actions.

Brainstorming completed its architectural path and root executed writing-plans
as the terminal next skill. The implementation plan is
`plans/2026-10-17-stage1-v5-dil-token-mean.md`. Root self-reviewed requirements
coverage, concrete steps, file ownership, types and placeholder absence.
There is one atomic implementation task because token sum, denominator,
contract and consumers must change together. SDD/task prompts, TDD, requesting
review and final verification/finish skills were read for their required gates.
No further method-selection question is needed; user already selected SDD.

Using-git-worktrees verified existing isolated branch `stage1_v5`: Git dir
`/root/workspace/ILVR/.git/worktrees/ILVR2`, common dir
`/root/workspace/ILVR/.git`, no superproject. Existing canonical venv is reused.
Fresh focused baseline command used the repository venv and
`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src`
with `python -m pytest tests/stage1/v5 -q`: **389 passed in 22.08s**, exit0.
Complete output: `logs/stage1-v5-dil-token-mean-20261017/baseline.log`.
No dependencies were installed or training launched.

## SDD atomic implementation task in progress

Implementer: `/root/dil_token_mean_implementation`, fresh context, base
`944a8fbba01fc3c0b856b3f940a08e0b9e9a09ba`. Root supplied the full atomic
task, concrete numerical/gradient/boundary requirements, approval identity,
file ownership, selected skills and all execution constraints. No other
implementation agent is editing in parallel.

Root directly read `red.log`: four numerical/repartition failures and
missing new window/scale APIs, total8expected failures, exit1. Root directly
read `green-reduction.log`:72passed12.19s exit0. This executes the planned
CPU first-nail witness with actual parent dispatch over four logical ranks;
it does not establish real distributed ZeRO execution.

Root directly read `green-contract-final.log`:111passed9.11s exit0.
Prior contract setup attempts and the two missing-runtime-denominator failures
are retained in their separate logs, including `contract-before-schedule.log`
and `red-runtime-contract.log`; they were not reported as green evidence.
Consumer tests and final integration are pending root verification/report.

Clarified root shorthand: no training/checkpoint mutation excludes real
training jobs and historical/model checkpoint overwrites. Established CPU-only
scalar unit fixtures remain authorized by the approved tests and canonical
AGENTS CPU-unit exception. Their fresh test-owned canonical
`checkpoints/stage1-v5-task7/<pytest-id>/...` outputs are allowed; no production
model, GPU training job or distributed launch is authorized. Baseline389
already included that existing fixture. Implementer was told not to exclude it.

## Implementation handoff and independent specification review

Implementer committed `94b049213cef83d3984f6f70dd224bc268393b5b` on
`944a8fbba01fc3c0b856b3f940a08e0b9e9a09ba`: 19 product/test/documentation
paths. Root inspected the actual diff and implementation report, including
the independently defined token objective, two scales, unchanged auxiliary
means, early training guards and schema-specific reconstruction.

Root read the full `green-integration.log`: **514 passed in 24.07s**, exit0,
for `tests/stage1/v5 tests/stage1/test_cli.py tests/stage1/test_composition.py`.
This is focused implementation evidence, not the final project-wide witness.
Both actual historical checkpoint-648 sidecars strictly read and reject new
training resume; inspection reported unchanged hashes. Fresh final inspection
will be run against the final reviewed source.

Fresh independent specification reviewer `/root/dil_token_mean_spec_review`
was dispatched read-only against `944a8fb..94b0492`, with exact approved scope,
call-path and falsifiable-evidence requirements. During review it identified
the stale cubic objective in the scalar verification consumer's restored
AdamW replay (`scripts/verify_stage1_v5.py`). Root inspected and confirmed the
remaining expression. Runtime impact and formal verdict are pending; no spec
pass, quality review or completion is claimed. Any fix will use the retained
implementation agent, followed by independent re-review.

Specification review formal verdict: **not yet compliant**, one P2 at
`scripts/verify_stage1_v5.py:505-509`. Independent arithmetic at `p=.2`
gave correct new-objective gradient `0.2259659469127655` versus stale replay
`0.1600000113248825`; AdamW first moments also differ. Small LR and broad
continuation tolerances let the existing test pass. This disproves the
implementation report's unqualified claim that the entire scalar witness
used the corrected oracle. That report is retained as the original handoff,
not accepted as proof for this finding.

Review artifact: `results/stage1-v5-dil-token-mean-20261017/spec-review.md`,
SHA256 `7dc06b2e1c6f4dec26fdb8c84f307b838753e9103b92eefcd5695117203497dc`.
Reviewer ran focused79 checks successfully and directly inspected the two
historical sidecars; those passing checks do not resolve the identified gap.
Root sent the exact finding to `/root/dil_token_mean_implementation` for
a narrow TDD fix plus optimizer-state regression and committed evidence.
Quality review remains gated on a later independent spec PASS.

## Specification fix and re-review

Same implementer committed the two-file causal fix
`95919e23d5faa5718ab9d20a2f270fac21a9aaf7`. The existing independent
literal token/block oracle now snapshots its parameter and deep-copied AdamW
state after each initial update, and restores those exact independent states
for each continuation. It no longer replays the obsolete cubic objective.
No numerical tolerance or parent execution protocol was changed.

Root directly read the complete fix diff and logs. A new independent literal
5-GT0/6-GT1-over-11-token calculation compares both restored oracle and
checkpoint AdamW moments before continuation. `fix-continuation-red.log`
fails precisely at `oracle exp_avg at step 1` on the previous consumer;
`fix-continuation-green.log` passes1 check in10.29s; complete protocol-file
integration passes79 in11.70s. These are CPU checks, not GPU execution.
The original implementation report is retained and its overstatement is
corrected additively by this fix evidence.

Root re-dispatched `/root/dil_token_mean_spec_review` for independent
delta/spec re-review of `944a8fb..95919e2`. Original finding evidence remains
in `spec-review.md`; new verdict will be in `spec-rereview.md`. Quality review
has not started and is still gated on spec PASS.

Independent spec re-review **PASS** at `95919e2`; P2 closed. Root read
the complete `spec-rereview.md`, SHA256
`f84fe5cae02a73812658ca14846a8765d46840bf96108371f366d02cb8ce9d18`.
Reviewer directly ran the causal regression (1 passed10.21s), observed both
moments at both saved steps, inspected deep-copy isolation and retained the
earlier unchanged-path evidence. This pass is bounded to static/CPU scope.

Only after that PASS, root dispatched fresh independent quality reviewer
`/root/dil_token_mean_quality_review` against full range
`944a8fb..95919e2`, using the SDD quality prompt plus requesting-code-review
template. It received the exact design/requirements, actual path-evidence
discipline, review/fix history and constraints. Quality verdict pending;
final whole-change review and project-wide verification remain required.

## Quality review passed; final review in progress

Quality reviewer **PASS** at `95919e2`, no Critical/Important/Minor findings.
Root read the full report `quality-review.md`, SHA256
`26f3e964c9c1f2c59b4c540df86d5d445e21b540cc81ba0e0bb3bccd24fd5392`.
Its fresh35 targeted checks passed10.55s; runtime MRO inspection resolved
actual inherited methods to the vendored Trainer and stated CPU/distributed
limits. The report preserves an earlier mistyped-node invocation as no tests
run, not passing evidence. No changes were requested.

Root then dispatched fresh independent final whole-change reviewer
`/root/dil_token_mean_final_review` against the same approved range
`944a8fb..95919e2`, with the full exact requirements, prior issue/fix history,
review artifacts and actual runtime-path/evidence instructions. This required
final review is separate from completed task spec and quality reviews.
Full-project verification remains reserved for root after that verdict.

## Final review passed; fresh completion verification

Final whole-change reviewer **PASS** at `95919e2`, no findings. Root read
the complete `final-review.md`, SHA256
`61a2a9ef94a0c4118ddb29fca216cb4718fa651320182755ab35566210df64ae`.
Reviewer independently inspected the full19-path change and actual
count/selector/parent/resume/verifier-state transitions. Its smallest causal
check passed4, deselected5, in10.18s; MRO assertions and diff checks passed.
No product changes occurred after the reviewed revision.

Finishing applies to this completed feature in the existing named linked
worktree. Root invoked and announced both finishing-a-development-branch
and verification-before-completion. Selected overlapping test-pass witness:
the project full suite `python -m pytest -q`, once on the reviewed revision,
using the established venv/PYTHONPATH/thread prefix. Distinct witnesses also
required: read-only real historical sidecar load/resume-reject/hash check,
reviewed-source equality, diff whitespace and frozen approval identity.
No passing final-suite or completion claim is made until full output and exit
status are read. Existing user-requested worktree is preserved; no integration
or cleanup action is authorized.

## Fresh verification completed and branch handoff

Reviewed product revision remains
`95919e23d5faa5718ab9d20a2f270fac21a9aaf7`. Root ran the exact selected
full-project command once, then read its complete output and exit status:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q
```

Result: **4896 passed, 11 skipped, 1 warning in 1232.54s**, exit0. The warning
is Torch C++ extension compilation's unset `TORCH_CUDA_ARCH_LIST`, emitted
by an existing Stage2 CPUAdam mapping test; no test failure. Eleven skipped
tests are not claimed as executed. Complete output:
`logs/stage1-v5-dil-token-mean-20261017/final-full-suite.log`, SHA256
`bd49388a25ab16c0a6cb9651db258fec4589e4369f0840603a7488b0d5e64a59`.
This one result is reused for verification-before-completion's test-pass
claim and finishing Step1; no identical broad rerun was performed.

Root also freshly executed the existing `inspect-historical.py` with the
same venv/environment prefix: exit0. Both actual schema1 checkpoint-648
sidecars strictly load; both training resumes reject with the explicit
block-mean/original-reviewed-code/fresh-run message. No model was loaded.
Legacy SHA256 before/after:
`2737109452dfdaaced55faa298f277457caa96c39f3cdc3e3f13dce42c769557`;
Adaptive before/after:
`8e064ae861790494e9f3142e49f39a95756eae3e579bb9792d2941beccfcf4f9`.
Fresh log `final-historical-inspection.log` and result
`results/stage1-v5-dil-token-mean-20261017/historical-inspection.json` both
hash to `23035a16841696250e65513a5aeb6b9bf7c0a62886c59818d40f6c9ab9b4d9cc`.

Claim-specific source/approval checks passed: product paths equal the final
reviewed HEAD (`git diff --exit-code HEAD -- README.md
scripts/verify_stage1_v5.py src tests`), approval spec retains the exact
authorized SHA256, and both working/range `git diff --check` commands exit0.
The product equality check was confirmed again after the full suite.
Only this root-owned control ledger is modified; pre-existing untracked
bytecode remains preserved. This final control-only commit does not change
the reviewed or tested product revision.

Finishing Steps2-3 were performed after the green suite: Git dir is
`/root/workspace/ILVR/.git/worktrees/ILVR2`, common dir
`/root/workspace/ILVR/.git`; named linked worktree `stage1_v5` is preserved.
Creation reflog binds its original HEAD to `148b0e8`; the matching feature
base is `Stage1-v5` (merge-base `148b0e8ead14456d02cc595f0cc32a0752ee22e8`).
The more distant `main` merge-base is `2497ee7`. No branch was switched.
Finishing Step4's four-option integration choice is the remaining human
handoff; no merge, push, discard, worktree cleanup or actual training launch
is authorized by this implementation completion. Until an explicit choice,
the requested branch and worktree stay in place.

Completion claims are limited to the implemented global-token DIL behavior,
independent reviews and stated tests/contract checks. No real multi-node
ZeRO gradient, production GPU resume, new trained checkpoint or generation
improvement has been established. A new token-mean training run must be
fresh and follow the standing separate launch policy.
