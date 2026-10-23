# Stage1-v5 mass weight: half entropy weight

## Intake and active phase

User request: `가중치는 entropy 가중치 * 0.5로 해놔라`.
Understood as: enable the discussed top-k mass KL for fresh Stage1-v5
DIL-BCE-ORIG training, with its omitted coefficient derived as half the resolved
top-k entropy coefficient; the current recipe must therefore resolve entropy
0.1 and mass 0.05.

Workspace: `/root/.codex/worktrees/1061/ILVR`, branch `stage1_v5`,
HEAD `ac223097cc5e3108af40a73af9723ac1caa7ec5a`.
This is an existing linked worktree; no checkout was created or switched.
The retained Subagent-driven method is evidenced by
`2026-10-17-stage1-v5-dil-token-mean-workflow.md:120-136`.
This request does not authorize updating the other inspected worktrees.

Active phase: bounded brainstorming, final short-design approval pending.
Skills executed: using-superpowers lifecycle routing; brainstorming context
inspection and bounded design; readchk context reconciliation (mass KL is the
resolved antecedent); hate design-commitment critique below. TDD and
using-git-worktrees instructions were read; implementation, tests, independent
review, final verification and finishing have not executed for this change.
Writing-plans is not triggered by this single bounded policy change.

Existing unrelated modifications are present, including network-only edits in
`docs/stage1-v5.md`; preserve them. The config, recipe and two test files below
match HEAD. No requested product change has been made.

## Frozen design proposal v1

- Scope: fresh Stage1-v5 configuration resolution and its existing managed
  training/evaluation recipe in this worktree.
- In `resolve_v5_config`, after alias canonicalization and only when `saved`
  is absent and mass was omitted, derive `lambda_topk_mass` as
  `0.5 * lambda_topk_entropy` (entropy omission resolves to zero). Existing
  numeric/type validation remains authoritative.
- Keep explicit mass overrides supported, including explicit zero. Resumes
  retain saved coefficients exactly and reject explicit conflicts as before.
  Do not change the dataclass's historical reconstruction defaults, loss
  formulas, reduction, saved schemas, Stage2 policies or checkpoints.
- Remove the recipe's explicit `--lambda-topk-mass 0` so fresh worker resolution
  uses the new rule. The recipe's entropy 0.1 thus resolves mass 0.05.
- Update the existing usage description and add focused behavioral coverage for
  the fresh ratio at more than one entropy value, omitted entropy, explicit
  override, saved mass-zero resume and actual recipe-to-worker resolution.
- Deliverable files: `src/stage1/v5_config.py`,
  `scripts/stage1_v5_train_eval.sh`, `tests/stage1/v5/test_config.py`,
  `tests/stage1/v5/test_launcher.py`, and `docs/stage1-v5.md`.
- Execution uses the retained Subagent-driven method, TDD, independent
  specification/quality review and fresh verification. No training is launched.

## Hate result for the same frozen proposal v1

```json
{"root":"A derived default is ineffective if the actual recipe or worker path still supplies an explicit mass=0; the behavioral claim depends on the emitted command reaching fresh config resolution without that stale override.","first_nail":"Use the real recipe dry-run and existing worker-argv construction to resolve the emitted Stage1-v5 config, asserting entropy=0.1 and mass=0.05 even after stale settings are sourced; separately witness that a saved mass=0 resume remains zero. This is proposed post-approval verification, not a training launch."}
```

First-nail status: not run; no implementation/premise experiment authorized by
this record. Static evidence: recipe line66 explicitly supplies mass0;
run_stage1_training.sh forwards mass only when its CLI option is present;
the saved contract validates an exact complete set of config fields.
No material irreversible premise is unresolved by this reversible configuration
change. Macrothink is not applicable: wrong-scope cost for this bounded change
does not materially exceed its fresh-reader overhead.

## Design commitment v1

Same design and scope as frozen proposal v1; no critique-driven revision.
Acceptance means fresh omitted-mass resolution follows entropy times 0.5 and
the real recipe uses it, while saved resumes and explicit overrides preserve
their represented coefficients. The launcher-override risk is addressed by the
recipe change and pending behavioral witness above. No material divergence or
unresolved disagreement is recorded.

Human approval of this exact commitment is pending. The prior approval of the
token-mean implementation does not authorize this coefficient-policy delta.
Only this controller-owned workflow record was created before approval.

## Approval and execution

The user replied `ㅇㅋ` to the exact short design commitment v1 above.
Approval binds the scope and coefficient relationship recorded in this file
at pre-approval SHA256
`d3d2605bb07b73a892609fed21405346b3ca17d4c3b9072c72d6bcd579e2ec03`.
Brainstorming is complete on its bounded path. The retained Subagent-driven
method applies to the single task brief below; no full implementation plan is
triggered. Training, checkpoint changes, other-worktree synchronization and
integration actions are outside this task.

### Global constraints

- Fresh omitted mass resolves as `lambda_topk_mass = 0.5 * lambda_topk_entropy`.
- Omitted entropy resolves to zero. Explicit mass overrides, including zero,
  remain supported.
- Existing resume coefficients are preserved exactly; explicit conflicting
  resume coefficients remain rejected.
- Do not change the dataclass's historical reconstruction defaults, loss
  formulas, reductions, saved schemas, Stage2 policies or checkpoints.
- Preserve all existing unrelated working-tree and index changes. In
  particular, preserve the pre-existing network hunk in `docs/stage1-v5.md`.
- Use only this repository's `venv/ilvr-venv` Python. No training, model loading,
  remote execution, new dependencies or subagents dispatched by workers.

### Task 1: Derive fresh mass weight and activate the managed recipe

Implement the frozen design commitment v1 in this worktree. Read the global
constraints above together with these exact requirements.

Files: `src/stage1/v5_config.py`, `scripts/stage1_v5_train_eval.sh`,
`tests/stage1/v5/test_config.py`, `tests/stage1/v5/test_launcher.py`, and
`docs/stage1-v5.md`. No unrelated product changes.

1. In fresh `resolve_v5_config`, after alias canonicalization, derive omitted
   mass as half the validated resolved entropy coefficient; entropy omission
   means zero. Preserve existing errors for unsupported/nonfinite/non-numeric
   coefficients; do not let arithmetic introduce incidental TypeError instead
   of existing validation errors. Keep the saved-config branch unchanged.
2. Remove the recipe's explicit mass-zero option so it reaches this policy.
   Existing entropy 0.1 must resolve mass 0.05 at the worker boundary.
3. Update the current usage description from mass0 to the half-entropy policy
   and selected mass0.05 objective. Preserve its unrelated dirty network hunk.
4. Follow TDD: first write and observe the expected failing behavioral tests,
   then implement the minimal fix and run the covering tests. Use hand-derived
   expectations for entropy 0.1 -> mass 0.05 and entropy 0.4 -> mass 0.2;
   entropy omission -> zero; explicit mass0 and a nonzero override remain exact;
   saved entropy0.1/mass0 stays mass0 on omitted resume options and rejects a
   conflicting explicit coefficient. Exercise the real managed recipe dry-run
   after stale settings are sourced, through coordinator/worker construction
   and real CLI parsing. No ranks or SSH may launch.
5. Run the focused config/launcher tests and smallest covering config/resume
   integration tests once after the change. Logs belong in
   `logs/stage1-v5-mass-half-entropy-20261023/`. Use
   `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src`
   with `./venv/ilvr-venv/bin/python -m pytest`.
6. Self-review and commit only the assigned task hunks. Do not stage unrelated
   network changes or the root-owned workflow record. Write a full report with
   RED/GREEN commands, exit status, complete log paths, files/commit identity,
   self-review and concerns; return the brief status contract to the root.

The root owns the single final full-project suite on the independently reviewed
revision and reuses it for verification-before-completion and finishing Step1,
per the standing combined-gate policy. Implementers/reviewers do not duplicate
that broad invocation. Full-suite success is not claimed by focused tests.

### Preflight task/interface scan

| Task / interface | Checked producer and consumer | Finding |
| --- | --- | --- |
| Task1 internal | fresh resolver / explicit and saved coefficient resolution | Same canonical names; fresh-only fallback leaves saved branch and explicit overrides intact |
| Task1 internal | recipe -> coordinator -> worker -> CLI config | Recipe explicit mass0 currently defeats fallback; remove it and witness real parsed worker config |
| Task1 internal | doc edits / pre-existing dirty network hunk | Header objective and network hunk are separate; stage only the task-owned header change |
| Task1 internal | tests / runtime scope | Dry-runs and CLI parsing require no training, remote launch or model load |

No conflicting tasks or extra interfaces exist. Bounded baseline is the two
directly affected config/launcher test files; final project-wide verification
will cover the reviewed working tree including preserved pre-existing changes.

### Execution evidence

Active phase: approved bounded implementation through Subagent-driven
development. Root read and executed the SDD setup/preflight routing and read
requesting-code-review, final-review and finishing requirements. Existing linked
workspace and empty initial index were verified. The focused pre-change baseline
passed **62 tests in 3.83s**, exit0, with complete output read from
`logs/stage1-v5-mass-half-entropy-20261023/baseline.log`.

Task1 implementer: `/root/mass_half_impl`, `gpt-6.1-sol` / `medium`, fresh context.
Brief/report/progress are scoped under
`.superpowers/sdd/2026-10-23-stage1-v5-mass-half-entropy-workflow/`.
Only approved requested-deliverable files may be edited by that implementer.
Independent task review, final review and final fresh verification are pending.

Task1 implementation is committed as
`e5123a059d42e261a2d916770e9e5d4ac5a4d448`
(`fix(stage1-v5): derive fresh mass weight from entropy`), exactly five assigned
files. Root read the complete implementer report and logs: RED exit1 with three
expected missing-behavior failures and 72 passes; GREEN exit0 with 128 passes,
14 deselections; additional two resume-boundary checks exit0 with two passes.
Logs: `logs/stage1-v5-mass-half-entropy-20261023/{red,green,green-resume-boundary}.log`.
No full-suite or launched-training result is inferred from these checks.

Independent task reviewer `/root/mass_half_review`, `gpt-6.1-sol` / `medium`,
reviewed `ac22309..e5123a0`: **spec compliant; quality Approved; no findings**.
Root read the complete review. The reviewer traced finite/type validation and
checked the named stale-mass environment risk at the unchanged CLI forwarding
boundary; no redundant tests were run. Controller resolved its two
cannot-verify items: the remaining documentation network hunk matches the
pre-change content, the committed file set is exactly scoped and requested code
matches reviewed HEAD; full-project verification remains the separate pending
completion gate. Task1 is complete at this task-scoped gate.

Final whole-change review dispatched to `/root/mass_half_final_review`,
`gpt-6-astra` / `high`, over the observed main merge base `ac22309` through
`e5123a0`, with the complete diff, brief, implementation report and task review.
No unresolved/deferred findings exist at dispatch. Final review and fresh
project-wide completion verification remain pending.

Final whole-change review: **Approved**, with no Critical/Important/Minor
findings, over the exact `ac22309..e5123a0` range. Root read the complete report.
The final reviewer independently traced actual parser provenance, the saved
contract route and the Trainer coefficient consumer. It inspected existing
focused evidence without rerunning tests.

Final reviewer declined-to-judge dispositions:

- Full-project success belongs to the controller's fresh command below and
  remains pending until its complete output and exit code are read.
- Remote/GPU-training participation, numerical behavior and accuracy are outside
  the approved non-training change. No such claims are made.
- Unrelated dirty network correctness is outside the submitted diff. Controller
  rechecked the initial versus current changed-path list and the documentation
  network hunk; the latter is unchanged apart from line-number shifts caused by
  the approved header edits. No unsupported all-files byte-for-byte baseline
  preservation claim is made.

Minimal raw evidence was preserved before any scratch cleanup in
`logs/stage1-v5-mass-half-entropy-20261023/`: task-1-brief.md,
task-1-report.md, task-1-review.md and final-review.md. SHA256, respectively:

```text
61b2afe842e52674c28383ae31742f4c998f883f7a455b5def4bdf599ab08c2f
a3bef43aed49bbd0db810a1c34bb2cadbf7055e083d17acf2a12fc8c6fcfef60
34e33436f32bffb713ca4b3c491915b2f491aeb6d7aabd2a2e71a440786f2da1
d3d2f9915a5fb74989c1d46eb40476c3c11d06424d6484521f2f2f56b81c55b6
```

Root invoked verification-before-completion and finishing-a-development-branch
together. Both local GPUs were idle with no compute processes immediately
before the selected single project-wide invocation on reviewed revision
`e5123a059d42e261a2d916770e9e5d4ac5a4d448`:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q
```

Complete output and exit witness are being written to
`logs/stage1-v5-mass-half-entropy-20261023/final-full-suite.{log,exit}`.
Result pending; full-project completion is not yet claimed. Root's distinct
scoped diff-whitespace, shell-syntax and reviewed-code identity checks completed
without output/errors. No identical broad suite was rerun.

### Interruption and workflow-cost question

The user interrupted the running turn and asked
`변수 하나바꾸는데 왜이렇게 헤비하게 하지?`.
The implementation and two independent approvals remain valid for `e5123a0`.
Root confirmed that unified session56590 is no longer available and the exact
owned pytest PID is absent. The log ends around71% without a final summary;
the shell's final exit-code witness was never written. Therefore the interrupted
invocation is not a successful full-suite witness and is not marked passed.

Root read diagnosing-superpowers for the specific workflow-cost question but
did not start transcript discovery, a diagnostic case, or its seven-analyst
workflow. The direct answer is limited to the observed current event: bounded
coefficient development was routed through required design approval, retained
SDD implementation, task review, final review and full-project finishing tests.
This is not a general diagnosis of the skill package.

No full-suite restart or additional diagnostic agents have been launched.
Full completion remains pending fresh full-suite evidence or an explicit user
exception for this task; no waiver is inferred from the cost question.

### Explicit exception and completion

Root asked whether to finish with the already passing focused verification
without restarting the interrupted full suite. The user replied `마무리해`.
This explicitly authorizes that task-specific full-suite exception. It does
not change the standing policy for future tasks or establish a full-suite pass.

Root freshly verified that the four assigned code/test files exactly match
independently reviewed product `e5123a059d42e261a2d916770e9e5d4ac5a4d448`, the
committed task diff has no whitespace errors, and the recipe passes `bash -n`.
A small pure-config runtime witness freshly confirmed entropy0.1 -> mass0.05,
entropy0.4 -> mass0.2, omitted entropy -> mass0, and saved mass0 remains mass0.
Output: `logs/stage1-v5-mass-half-entropy-20261023/completion-check.log`, exit0.
Root reread the complete focused GREEN logs: 128 passed and two additional
resume-boundary cases passed. Both independent reviews remain approved without
findings. The interrupted full suite is neither restarted nor marked passed.

Requested change is complete under the user's explicit verification exception.
The existing `stage1_v5` branch and workspace are preserved; no merge, push,
remote synchronization or training launch is requested or performed. All
controller decisions were already in the approved design or this explicit
exception; there are no parked findings or undisclosed scope rulings.
The minimal task brief, implementation/review reports and execution logs remain
in the existing task log directory before scoped SDD scratch cleanup.
