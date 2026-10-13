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
