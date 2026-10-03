# DIL GT-conditioned top-k task brief and workflow evidence

## Intake and approval

Understood as: add a selectable GT-conditioned rejected objective, sharing one
OFF-selected TopK union GT token-ID set between rejected, OFF mass and clean
mass; preserve both existing top-k variants and historical resume semantics.

Initial path: bounded change to the existing DIL execution flow. After the user
challenged the missing spec/plan (`spec/plan은 어디에 팔았지?`), the controller
upgraded to the full architectural spec/plan path. The mathematical intent is
approved in chat; the written-spec approval was subsequently supplied by the
explicit writing-plans invocation recorded below. Do not use
the earlier bounded classification to omit the plan or written-artifact gates.

User approval evidence in this conversation:

- Requested `exp(GT) / sum(exp(Top-k union GT))`, preserving the old options.
- `유지로 진행해` selected mean token log-probability before rejected softplus.
- `mass도 top-k \\union GT와 맞춰야하는건 당연한 설계잖냐 뭘 니가 추가해`
  clarified that the shared union for mass is inherent in the requested design.
- After the assistant specified that a single OFF-selected union is reused for
  rejected and both masses (no separate clean top-k), the user said
  `그렇게 진행해라.` This approves the reconciled bounded design below.

Earlier assistant proposals to retain the old mass support, or to treat union
mass as an independent new feature, were incorrect interpretations and are
superseded. No product code was changed under those proposals.

## Approved mathematical behavior

- New selector: `dil_bce_rf_topk_gt`. Existing selectors retain their meanings.
- For each target position t, choose K_t from detached OFF logits, and construct
  S_t = K_t union {GT_t}, deduplicated. All three calculations reuse these IDs.
- log q_GT,t = OFF_logit_GT,t - logsumexp(OFF_logits[S_t]). Selected logits,
  including numerator and denominator, remain differentiable.
- Per original step, rejected = softplus(beta * mean_selected_t(log q_GT,t)).
  There is no q-weighted candidate expectation and no tokenwise softplus mean.
- Chosen retains the original DIL positive: softplus(-beta * mean_selected_t
  clean full-vocabulary GT log-probability).
- M_clean,t and M_OFF,t are full-vocabulary probability sums over the same S_t.
  Mass = KL(Bernoulli(stopgrad(M_clean,t)) || Bernoulli(M_OFF,t)). It matches
  total set mass, not its internal token distribution.
- Preserve entropy-position selection, token-then-original-step mass averaging,
  original-step denominators, zero contribution for steps with no selected
  positions, and sample/distributed weighting.
- If S_t spans the full vocabulary (possible with k=V-1 and GT outside K_t),
  both masses equal one and mass KL and its gradient are zero. Do not pass an
  empty complement through the existing group-gap helper.
- Preserve k validation (2 <= k < V), existing options and saved coefficients.
  New DIL defaults remain beta=1, margin=0, preference weight=1, CE weight=0.1.
  Keep the existing independently configurable mass coefficient; this task does
  not authorize changing its global default or launching a training run.
- Save a distinct formula identity and reject cross-objective resumes.

## Deliverable and acceptance boundaries

Connect the new selector through the existing loss/configuration, checkpoint
identity, trainer/sidecar and coordinator/CLI paths. Document the selectable
objective. Avoid unrelated refactoring and new orchestration.

Verification must distinguish GT already in K from GT outside K, deliberately
different clean/OFF top-k sets, live denominator and detached clean gradients,
step reduction order and empty selections, extreme finite/BF16 logits, full
union support, objective defaults, coordinator forwarding, resume identity,
and unchanged legacy variants. Use independent numerical oracles and the
smallest critical existing sidecar/trainer integration check.

## Current workflow state

- Base revision: `2f48c51985c395c16bea80a89847d7586728f9cd`.
- Existing worktree: `/root/workspace/ILVR/.worktrees/dil-loss-defaults-20261001`.
- Initial status: only untracked `src/**/__pycache__/` directories, preserved.
- Skills executed: using-superpowers intake; brainstorming context exploration,
  clarification, bounded design and user approval.
- TDD and requesting-code-review instructions read for the next phases; no
  implementation, red/green cycle, independent review or final verification has
  run for this new objective yet.
- Execution method: awaiting user selection. The earlier method-choice question
  was not answered; do not infer native or subagent implementation.
- Pending: written-plan review and execution-method selection, implementation/TDD,
  independent specification and quality review, fixes, fresh verification and
  applicable branch finishing.
- No agents dispatched. No training, checkpoint modification, merge or push.

Append actual commands/results, agent IDs, reviewed revision or diff hashes and
final verification witnesses as work proceeds. This brief is a controller
artifact, not implementation completion evidence.

## Routing correction and recovery

The full path requires written-spec review before writing-plans, followed by
plan review/method selection before implementation. Actual earlier action was
a bounded classification, in-chat design clarification and this task brief;
no written spec or plan existed and no product implementation had started.
The controller's earlier mass-support interpretation was corrected in chat
before the user's final mathematical approval. Remaining uncertainty is
written-artifact approval and execution method, not the shared union contract.

Recovery: write the approved design as a spec, self-review its requirements and
scope, commit the control artifacts, and request review of that concrete spec.
Then create the implementation plan under writing-plans. Later review does not
retroactively claim the earlier missing artifacts existed.

Written-spec self-review: read the entire spec and checked support identity,
loss signs and averaging, detached/live gradients, empty/full-support cases,
legacy compatibility and verification scope against the inspected production
paths. Made the conditional-score diagnostic name/reduction explicit. Ran
`rg -n 'TBD|TODO|FIXME|placeholder'` on the spec; no matches. No product tests
were run for these two controller documents, and implementation remains
unstarted. At that point the next gate was user review of the written spec.

## Writing-plans authorization and outcome

After the written spec was linked and committed at `1c3eae3`, the user explicitly
invoked `$superpowers:writing-plans`. This is the instruction to proceed from
that spec into planning; no repeated spec approval was requested. The invocation
does not select an implementation method or approve a plan that did not yet exist.

Skill executed: writing-plans (scope/file mapping, three TDD task boundaries,
exact interfaces, runtime witnesses, self-review, execution handoff). Artifact:
`plans/2026-10-03-dil-topk-gt.md`. Active phase is written-plan review and method
selection. No product code, tests or training were executed in this planning turn.

Self-review compared all nine spec sections to tasks, checked seven-field GT
statistics and the ten-value sidecar layout, mapped the five additional review
risks to tests, and checked command/file references against the worktree. Found
that the existing full-logit test oracle initializes unweighted CE; the plan now
explicitly makes that oracle use lambda_CE for the fresh-default GT cases.

Document verification: repository venv Python checked mandatory plan headings,
exactly three tasks, no TODO/TBD/FIXME markers, and all 20 referenced existing
source/test/document paths (two new test files are explicitly planned). Result:
exit 0. `git diff --check` also returned exit 0. These checks verify the planning
artifact, not the unimplemented feature. The plan covers one objective pipeline,
not independent sub-projects; no plan-review subagent was dispatched.

## Execution authorization (current)

User `서브에이전트로해` approved the presented plan at 35e169d and selected SDD. Earlier awaiting-selection entries are historical. Active phase: Task 1 implementation after baseline. Using-git-worktrees confirmed the existing named linked worktree; SDD ledger is `.superpowers/sdd/2026-10-03-dil-topk-gt/progress.md`. Preflight producer/consumer and within-task scan is recorded there; no conflicts. Baseline command and log are recorded there: 17 passed in 2.90s, exit 0. No training launch.

## Task 1 implementation and independent gate

Implementer `/root/gt_task1` (gpt-6.1-sol/high), commit `9c5d0f9`, added GT token
statistics and shared stable mass handling. RED: six expected missing-feature
assertion failures. GREEN: new/old primitive suite 20 passed. Independent reviewer
`/root/gt_review1` (same model tier) approved spec compliance and quality with no
Critical/Important/Minor findings; report at the task workspace `task-1-review.md`.

The first full run was NOT green: 16 failed, 3947 passed, 38 skipped, 33 errors
(task1-full.log under logs/dil-topk-gt-impl-20261003T074736Z). Root diagnosed its
own plan's uppercase timestamp and hidden CUDA as causes; controller correction
`6cb832e` fixes the control document, not product code. Existing timestamp test
then passed, and the exact 49 failed/error nodes passed with GPU0 visible in
12.26s. This causal subset is not reported as a full-suite pass. Both GPUs had
0 MiB/0% utilization before that unit-test run. No training job was launched.

Ruling: correct plan timestamp to lowercase and expose an idle local GPU for
existing CUDA unit tests — those failures came from controller-supplied execution
settings — cost if wrong is resource contention, mitigated by rechecking idle
resources before any unit-test use. New GT tests remain CPU. Final fresh suite
against the reviewed completed revision remains pending.

Task 1 gate complete; Task 2 delegated to `/root/gt_task2` at base `9c5d0f9`.

## Task 2 implementation and independent gate

Implementer `/root/gt_task2` (gpt-6.1-sol/high), commit `8a35ffa`, connected
selector/defaults, typed reduction and identity. RED: eight expected missing
selector failures. GREEN: 81 passed after correcting a fixture's unused
full-vocabulary diagnostic gradient assumption (the first green attempt was
1 failed/80 passed). Full suite: 4031 passed, 11 skipped, one existing CPU Adam
compilation warning in 1216.05s, exit 0. Logs: task2-red.log,
task2-green.log, task2-green-fixture-corrected.log, task2-full.log under the
same execution run. Root read the final suite output and committed file stat.

Independent reviewer `/root/gt_review2` (gpt-6.1-sol/high) approved spec and
quality for 9c5d0f9..8a35ffa, with no Critical/Important findings. Minor finding:
TORCH_CUDA_ARCH_LIST was unset, so compilation targets were implicit. Resolve
future suite settings using the observed local architecture without repeating
the completed suite. Actual runtime/replay/distributed/parent-resume verification
is explicitly assigned to Task 3. Active phase: Task 3 implementation.

## Task 3 implementation evidence

Implementer `/root/gt_task3` (gpt-6.1-sol/high), commit
`6b222a64ad04633068d8b83b378e78a010500309`, connected runtime, worker, resume,
planned-stop and documentation paths. Twelve assigned files changed; generic
coordinator forwarding already passed both-worker witnesses and stayed unchanged.
Runtime RED: 14 failed/65 passed. Intermediate runtime: 1 failed/78 passed
(the remaining shell selector). Routing RED: 5 failed/68 passed, including one
invalid old-topk test fixture corrected without a product change; selector-switch
causal check then 2 passed. Final focused tests: runtime79 passed, lifecycle73
passed, both warning-free. Shell syntax and diff checks returned exit0.
Full suite: 4060 passed, 11 skipped, 1 warning in1220.97s, exit0. Root read complete
full-suite output and committed file list. Logs and numeric exit files use
`task3-runtime-red`, `task3-runtime-green-1`, `task3-runtime-green-2`,
`task3-runtime-final`, `task3-routing-red`, `task3-routing-red-fixed-test`,
`task3-routing-green`, `task3-shell-syntax`, `task3-diff-check`, `task3-full`
under the same execution run's logs directory.

Setting TORCH_CUDA_ARCH_LIST=9.0 did not eliminate the CPUAdam JIT warning:
installed DeepSpeed op_builder/builder.py:570-572 saves and blanks that variable
inside jit_load, restoring it at606-607. No dependency edit or warning suppression
was applied. The warning remains visible and is not a pristine-output claim.
Task3 review is assigned to `/root/gt_review3`; final whole-change review and fresh
verification remain pending. No training/remote-participation/performance claim.

Task3 independent reviewer `/root/gt_review3` approved specification compliance
and quality for8a35ffa..6b222a6. No Critical/Important issues. Review traced actual
trainer->sidecar->typed-reducer paths and pre-parent resume validation; independent
oracle, replay and artifact witnesses match the spec. The reviewer independently
confirmed DeepSpeed's environment clearing, retaining this Minor as unresolved.

Ruling: retain the existing DeepSpeed JIT architecture warning as a reviewed,
unresolved environment finding without dependency edits or suppression — the
dependency temporarily clears the explicit setting and all tests passed — cost
if wrong is a separate extension-compilation compatibility fix. Do not claim
warning-free output. All three task gates are complete; final broad review next.

## Final independent review and completion-gate setup

Reviewer `/root/gt_final_review` (gpt-6-astra/high) reviewed2f48c51..958c677
against spec/plan, actual CLI->trainer->sidecar->primitive->typed-reducer paths,
checkpoint replay and pre-parent identity validation. Verdict: Ready to merge Yes
for this change; no Critical/Important defects. Existing DeepSpeed warning remains
an unresolved Minor, not a clean-output assertion. Reviewer read full4060/11/1
and focused79/73 logs without rerunning suites. Final verification is still pending.

Ruling: accept the final review's four explicit scope exclusions — real two-node
GT training/GPU GT replay/distributed optimizer restore; convergence or performance;
pre-baseline equivalence with other branches; dependency remediation — each is
outside the approved implementation scope. Cost if those capabilities are later
needed: their own training/compatibility/environment validation is still required;
none is inferred from these tests.

Active gates: verification-before-completion and finishing-a-development-branch.
Claim witnesses selected before execution: one fresh project full suite for the
shared test-pass claim; bash syntax for the launcher; diff check for patch hygiene;
independent spec/call-path review plus the suite's causal tests for requirements.
No identical second final suite will be run. Product revision6b222a6 and reviewed
head958c677 have identical product content; subsequent controller records only
append workflow evidence. Branch fork recorded atf575cd9, current tip of
feat/dil-bce-port; no merge/push/cleanup decision inferred from this observation.
