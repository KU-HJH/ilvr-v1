# DIL-BCE-ORIG workflow evidence

## Intake and approved scope

Understood as: implement the token-level chosen/frozen-reference BCE objective
and optional V-off top-k entropy described in the user's
`documents/dil-bce-orig-v1.md`, in a worktree named `dil-bce-orig`, preserving
existing objective and resume semantics.

The selected Superpowers path is architectural: the work adds a frozen model
lifecycle, same-clean-history replay, and reference artifact/resume contracts.
This is implementation work, not authorization to launch training.

The controller presented section 1: base `feat/dil-bce-port` at
`00c212cb598d88e5cfe22a40d11d320f04c609d5`, a separate frozen reference loaded
from a checkpoint, replay of detached student latent inputs with reference-owned
token/image processing and KV, and the full requested loss/entropy/mass scope.
The alternative discussed was sharing the base and switching adapters.
The user replied: `그렇게해라 worktree 경로가 어디지?`
This approves that section and workspace setup. It does not approve a written
specification or implementation plan that has not yet been presented, nor select
an implementation method.

## Workspace and inputs

- Workspace: `/root/workspace/ILVR/.worktrees/dil-bce-orig`.
- Branch: `dil-bce-orig`.
- Base: `00c212cb598d88e5cfe22a40d11d320f04c609d5`.
- Root checkout was not a linked worktree; `.worktrees` was already ignored.
- Native worktree creation was inspected; it accepts a ref but no destination or
  name. The user's approved exact path was honored with
  `git worktree add .worktrees/dil-bce-orig -b dil-bce-orig 00c212cb598d88e5cfe22a40d11d320f04c609d5`.
- Existing repository venv, data and checkpoint locations are reused through
  symlinks; no package installation or environment replacement occurred.
- The canonical input document was copied without modification into this
  worktree. SHA-256:
  `9b26bd40ec40dbf882e7254f0b650595538d2432e7ace2fe363dbe8ecaf1eed1`.
- Canonical and worktree AGENTS instructions were read. Future training must
  separately satisfy the canonical training-defaults and multi-node policies.

## Executed phases and remaining tasks

1. using-superpowers: read and applied intake and lifecycle routing.
2. readchk: compared requested document with actual repository and branch history.
3. brainstorming: architectural classification, intent reflection, repository
   exploration, two reference approaches, and section-1 approval completed.
4. using-git-worktrees: checked isolation, created the requested workspace,
   reused existing environment/storage, and ran the focused baseline below.
5. verification-before-completion: read for workspace/baseline claims. Branch
   finishing does not apply at this design stage; no feature completion claim.

Active phase: brainstorming, remaining design sections and repository/spec gaps.
Pending in order: section confirmation; written spec and self-review; frozen
proposal and root-only hate plus applicable macrothink critique; exact design
commitment and written-spec approval; writing-plans; written-plan review and
user-selected implementation method; TDD implementation and independent reviews;
fresh claim-specific verification and applicable branch handoff.

No implementation/reviewer agents have been dispatched. No product code has
changed. No training, remote execution, merge, push or feature tests occurred.

## Observed repository differences for the remaining design

- `src/step_preference/dil_config.py` currently allows entropy selection sources
  `clean` and `voff`; `gap` and `intersection` from the input document are not
  established features of this base.
- The base config has no `preference_topk_mode`/`sync_topk` field. These must be
  treated as new work for the new objective, not claimed as existing behavior.
- Actual identities are Stage2 PEFT schema 4 and DIL identity schema 1, not the
  input document's assumed schema 21. Choose a compatible versioned extension
  for the real identity format in the design.
- The clean model forward currently exposes final input embeddings and can
  return latent states. This is static evidence of useful data, not evidence
  that the required reference replay API already exists or is correct.
- The actual sample-weight wrapper zeros padded samples after objective
  evaluation. New global-step reduction must exclude their steps from its
  denominator while retaining required collective ordering and graph support.

## Fresh baseline witness

From this worktree, at the unchanged product revision above:

```bash
export PYTHONPATH="$PWD/transformers/src:$PWD/src"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
./venv/ilvr-venv/bin/python -m pytest -q \
  tests/step_preference/test_dil_bce_loss.py \
  tests/step_preference/test_dil_topk_loss.py \
  tests/step_preference/test_dil_config.py \
  tests/step_preference/test_dil_trainer.py \
  --basetemp="$PWD/checkpoints/dil-bce-orig-design-20261013/baseline"
```

Observed complete output: `116 passed in 9.70s`, exit 0.
Log: `logs/dil-bce-orig-design-20261013/baseline.log`; numeric exit:
`logs/dil-bce-orig-design-20261013/baseline.exit`.
This is an existing DIL baseline, not a full-suite or new-feature pass.

## Design-section approval and frozen proposal v1

The controller presented token-before-step full-vocabulary BCE with independently
weighted entropy/mass, immutable reference snapshot and objective-safe resume,
and numerical/gradient/history/DDP/legacy verification. Asked whether to confirm
these contracts and write the specification, the user replied `진행해`.
This completes the preliminary section confirmations and authorizes writing the
specification. Final written-spec approval and implementation method remain pending.

Written proposal: `specs/2026-10-13-dil-bce-orig-design.md`, proposal v1.
SHA-256: `add66ad6e3e9c517426def6a5a24108e5b70a2bbcd8e6579b8dbcb4fa9f7e972`.
Self-review checked the equations, forward ownership and ordering, detached/live
boundaries, config/resume routing, snapshot storage, real schema versions and
verification scope. Explicit reconciliation decisions cover the absent selection
modes/replacement backend and the zero-loss padded accumulation tail.
Document checks: ten sections and required invariants present; source-document
hash matches; no TODO/TBD/FIXME/placeholder matches; `git diff --check` exit 0.
No source-code tests were added or run for this documentation step.

The load-bearing design spans reference lifecycle, modified model replay and
distributed objectives. Solving the wrong conditioning/reduction problem would
cost substantially more than three bounded read-only fresh reads. Therefore the
standing macrothink condition is satisfied. Root will execute hate and then
three same-model fresh reads of this frozen proposal; agents cannot invoke
Paperthin hooks or start implementation. The proposal remains unchanged until
all critiques have returned.

## Hate result for proposal v1

```json
{
  "root": "The density ratio is only the specified objective if the frozen model evaluates exactly the student's externally supplied latent history while computing ordinary token/image representations with its own weights. A replay that silently regenerates latents or reuses the student's entire embedding tensor can pass shape and detached-gradient checks while optimizing a different objective.",
  "first_nail": "Run one tiny patched-Qwen CPU replay experiment with two latent blocks: capture the student's actual latent slots and positions, rebuild all other inputs through an independent reference, and compare reference logits to a literal reference-decoder evaluation. First require equal-weight eval agreement; then change only reference token/image embedding parameters and require non-latent representations to change while replayed latent slots remain exact. Check there is no student KV reuse and no reference path to student gradients. This needs no real checkpoint or training run."
}
```

First-nail status: proposed, not executed. Existing baseline tests are not this
witness. Because the probe creates experimental replay code, separate explicit
authorization is required before running it; design-section approval is not
probe authorization. If it is not run before final commitment, the same-history
premise remains explicitly unverified and requires a human risk waiver under
the standing design-approval policy.

The first staged whitespace check flagged four existing Markdown hard-break
lines (two trailing spaces) in the byte-identical user source document. No
source text was rewritten to satisfy that style check. Authored control
documents are checked separately with `git diff --cached --check --
docs/superpowers`; the source copy is checked by byte comparison and SHA-256.
This is an input-format exception, not a product/runtime defect or a test pass.
The CPU first-nail experiment was requested separately through asynchronous
user input; it remains pending until an explicit answer arrives.

## Macrothink execution and synthesis

Frozen proposal v1 was committed at `de7999256728d0ef832ded690e36a60371cbfe42`.
Root dispatched three fresh-context, same-model, read-only agents under the
standing design-commitment policy: `/root/orig_fresh_read_1`,
`/root/orig_fresh_read_2`, `/root/orig_fresh_read_3`. Each was instructed to record
an independent framing from the stripped problem before opening the frozen
proposal, verify its SHA-256, avoid other critiques and perform no experiments,
edits, hook invocations or further delegation. All three reported those initial
framings before reviewing v1 and verified the supplied proposal hash.

The stripped problem retained token imitation, full-vocabulary frozen-policy
negatives, online student latents, optional ablation uncertainty, legacy/CE/resume
preservation and fixed global16/tail policies. It omitted the chosen class/API
names, separate-model recommendation, loss name and the controller's first-nail
objection. The reads independently emphasized score/link, positive/reference
overlap, conditioning/differentiation, weighting and recoverable effective policy.

Classification against the current direction:

- Incompatible divergence: none identified in these static reads. This is not
  evidence of runtime correctness or a majority-vote approval.
- Read 1: divergent-compatible. Require actual parent Trainer/DeepSpeed tail
  scaling and collective-order witnesses; use a literal reference decoder as
  the replay comparator, rather than general student/reference equality.
- Read 2: divergent-compatible. State the conditional semi-gradient and changing
  reference outputs; document grouping-dependent weighting, test the actual
  padding wrapper, and distinguish eval identity from training-mode dropout.
- Read 3: divergent-compatible. Scope support-only OFF gradients to entropy,
  preserve complement gradients for mass, add actual GAS4/tail evidence, and
  hold reference logits/discrete selectors fixed during finite differences.
- Convergent portions of all reads: density-ratio formula, independent
  reference-owned embeddings and immutable reference identity were consistent
  with their initial framing. Convergence is reassurance only, never proof.

Root clustered the compatible findings into (1) exact conditioning/gradient
boundaries and (2) actual distributed execution versus reduction algebra. The
final design retains the architecture and addresses these in sections 3/4/6/9.
No material disagreement is relabeled as resolved by votes. Known production
and PEFT uncertainties remain explicit acceptance gates. Root alone performed
synthesis; these agents did not provide implementation reviews or authorize a
phase transition.

## Explicit probe authorization and observed first-nail evidence

The user answered the separate experiment question with
`CPU 재생 검사 실행 승인`. Root then created one throwaway experiment in
`logs/dil-bce-orig-premise-20261013/probe.py` and ran it once using:

```bash
export PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
./venv/ilvr-venv/bin/python logs/dil-bce-orig-premise-20261013/probe.py
```

No model artifacts were loaded or saved, no optimizer step or training run was
executed, and no product implementation was added. The fixture is a randomly
initialized native tiny Qwen, length17, two latent blocks, latent positions
6/7/12/13, predictors8/9/14/15. A copied model supplies equal initial effective
weights for this experiment only; it is not the production reference loader.

Observed exit0/pass; the controller read complete stdout/JSON and numeric exit:
equal-weight eval logits and literal-reference replay logits each had max error0;
changed reference token/image representations differed0.15/0.20 while captured
latent slots remained exact. Negative controls differed0.7284281849861145
(all-student-embedding logits) and3.5637803077697754 (reference-rerolled latents).
Two observed reference replays made one decoder call each with no student KV.
Reference outputs were detached from a requires-grad latent leaf, student
backward retained nonzero gradients, and reference parameters/buffers stayed
unchanged after the intentional reference perturbation and subsequent replay.

This establishes a narrow CPU feasibility witness, not the production replay
API, PEFT behavior, new objective, optimizer exclusion, distributed training,
snapshot/resume or GPU capacity. The first nail did not disprove the premise;
there was no reason to stop the remaining advisory reads or redesign the
architecture on that basis. No unexecuted experiment is claimed as evidence.

Preserved evidence:

- Probe source SHA-256: `391fda6285789cdb57d8aabc2fb4c8a94fa3f076597f5eab0bcbaf6d20308c07`.
- Results: `results/dil-bce-orig-premise-20261013/result.json`.
- Log and exit: `logs/dil-bce-orig-premise-20261013/probe.log`, `probe.exit`.
- Result and log SHA-256 (identical serialized output):
  `cfcb4d8153fc4e9b252e18e4bf8660efa318a0114a01cc9724ce940eb4195899`.

## Current phase: final written-spec approval

`specs/2026-10-13-dil-bce-orig-design.md` is now design commitment v1, with
agreed intent/scope, final design, exact changes from frozen proposal v1,
first-nail evidence, advisory divergences and remaining uncertainties. All
preliminary design sections, self-review and the mandatory root advisory hooks
have actually executed; this is not implementation completion.

The source document and feature code remain unchanged. The controller has
authored only control documents and the explicitly authorized throwaway probe.
No written-spec approval, plan approval or implementation-method choice has
arrived yet. Writing-plans is the next phase only after written-spec approval;
implementation skills, TDD/product edits, independent code reviews and branch
finishing have not started. Branch finishing is not the next gate here.

The final control-document check initially used a raw substring assertion for
a sentence split by a Markdown line break; that assertion failed. Its correction
normalizes whitespace before checking the same semantic text. This is a
verification-script formatting correction, not a changed design or product fix.
The final successful check and commit are recorded by the subsequent command
output/Git identity; no failed assertion is treated as a pass.

Final document verification returned exit0: eleven complete design sections and
the review clarifications are present; the canonical input is byte-identical;
probe source/result hashes and numeric exit match the observed witness; no
product-code diff exists from the approved base. `git diff --check` returned0.
Log: `logs/dil-bce-orig-design-20261013/final-doc-check.log`.
Final commitment SHA-256:
`065e24b0d953eceddfa173ff842d3e8259f57861d233e2cf3fa72297ed936f52`.
The commit containing this record and that exact specification is the written
design handoff identity; approval is still pending and no plan was written.

## Written-spec approval and implementation planning

After the controller presented design commitment v1 and requested final written
specification approval, the user replied `계속`. This is the actual approval
event for the complete written spec at commit
`112b09c230250e3e3adcaec77ea6171701b2032d`, SHA-256
`065e24b0d953eceddfa173ff842d3e8259f57861d233e2cf3fa72297ed936f52`.
It approves sections 1-11, including the recorded compatibility decisions,
conditional replay, reduction/padding convention and runtime evidence boundary.
The earlier pending-status statements remain historical records. The approved
spec bytes have not been edited. This approval does not choose an implementation
method, approve a plan not yet presented, or authorize training.

Root completed brainstorming's approved architectural handoff and read/applied
`superpowers:writing-plans`. Plan:
`plans/2026-10-13-dil-bce-orig.md`. The six task boundaries are numerical kernels,
configuration/identity, history replay, effective reference lifecycle,
Trainer/distributed reduction, and entrypoint/artifact/runtime-witness integration.
Each specifies files, consumed/produced interfaces, failing assertions, RED/GREEN
commands and selected-method review/commit gates. CPU/Gloo/actual-parent evidence
is separated from the unexecuted two-node/four-rank production witness.

Root's inline writing-plans self-review checked all eleven spec sections and
source D1-D8/E1-E7/N1-N4/I1-I12, cross-task types, exact CLI field names, actual
existing file locations, five concrete review-focus tests, and proportionality.
It made explicit the CLI provenance field and Task 5's modifications to the
shared statistics/output modules, and expanded abbreviated test invocations.
No planning subagents were used. The pre-rank snapshot preparation is a concrete
implementation of the already approved one-writer-per-node rule; no rank-side
wait/retry system or new training launcher is planned.

The controller read TDD, requesting-code-review and finishing-a-development-branch
to account for their future gates; their implementation/review/finish phases
have not executed. verification-before-completion is applied now only to the
control-document handoff: plan structure/coverage, unchanged approved spec and
source hashes, authored whitespace, and absence of product-code changes. Branch
finishing is not the next gate, so no project full suite is appropriate for
this planning-only claim. Existing baseline/probe results are not new tests.

Active path: architectural; current phase: plan review and implementation-method
selection, pending the user's response to the plan handoff. The recommendation
is Subagent-driven because math/replay/artifact/DDP interfaces can silently
change the objective; only the user can select it or Native. Product changes,
TDD execution, implementation/reviewer agents, final independent review,
production execution and branch finishing remain unstarted.

Fresh plan-document verification returned exit0. The controller read the full
output: six tasks and their interfaces/TDD/review gates, all five named
review-focus tests, preserved approved spec/source hashes, canonical source byte
comparison, existing integration paths, authored whitespace and no product-code
changes all passed. This is documentation validation only, not feature testing.
Log: `logs/dil-bce-orig-plan-20261013/document-check.log`.
Plan SHA-256:
`e699acb7432bb4d116ab1b9f64c7e911ddbe45026ed4593d3f27ea92b5310722`.
The commit containing this plan and approval/evidence append is the plan handoff
revision. Plan approval and implementation-method selection are still pending.

## Plan approval and Subagent-driven execution

The user replied `Subagent` to the presented plan and execution-method choice.
This approves plan commit `e597354f09b9d4f2db4925cb48ff55e9157f8da4`
and selects `superpowers:subagent-driven-development`. The approved plan hash
is unchanged. Root created this plan's scoped SDD workspace, checked the
existing worktree, and completed the required task/interface preflight table.
See `2026-10-13-dil-bce-orig-implementation.md` for the durable ledger,
including the one test-order ruling. Task1 implementation dispatch is next;
no implementation is yet claimed. Training authorization remains separate.
