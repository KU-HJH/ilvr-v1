# DIL GT-conditioned top-k task brief and workflow evidence

## Intake and approval

Understood as: add a selectable GT-conditioned rejected objective, sharing one
OFF-selected TopK union GT token-ID set between rejected, OFF mass and clean
mass; preserve both existing top-k variants and historical resume semantics.

Initial path: bounded change to the existing DIL execution flow. After the user
challenged the missing spec/plan (`spec/plan은 어디에 팔았지?`), the controller
upgraded to the full architectural spec/plan path. The mathematical intent is
approved in chat; the written spec still requires its own approval. Do not use
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
- Pending: user review of `specs/2026-10-03-dil-topk-gt-design.md`, writing-plans,
  written-plan review and execution-method selection, implementation/TDD,
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
unstarted. The next gate is user review of the written spec, not implementation.
