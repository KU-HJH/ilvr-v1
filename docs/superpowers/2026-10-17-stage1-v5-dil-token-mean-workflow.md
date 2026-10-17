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

Active lifecycle: final commitment v1 approved; architectural brainstorming
complete, writing-plans completed, retained user-selected Subagent-driven
implementation next. Earlier pending entries below describe their time; the
approval section at the end is authoritative for current authorization.

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
