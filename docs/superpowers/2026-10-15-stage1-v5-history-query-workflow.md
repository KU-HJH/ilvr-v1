# Stage1-v5 history-query intake

## Authority and current phase

Understood as: create a separate `stage1_v5` worktree, inspect canonical
`documents/ILVR_History_Query*` and the current Stage1-v5 target-query path,
then report a proposed option name before making product changes.

The initial investigation and naming report was delivered. The subsequent
user instruction sets `history_text_gamma=0.9` and requests the next stage.
The user has also confirmed the recommended architecture section below.
No complete written design, implementation plan/method, premise experiment,
training launch, or implementation has been approved. Existing retrieval
brainstorming in the separate `Stage1-v5` worktree does not authorize this change.

Active phase: writing-plans; plan review and execution-method selection pending.
Latest explicit naming decision: `--query_mode` replaces the initially
recommended `--history_query_mode`; proposed values remain legacy/adaptive.
An implementation would follow the architectural path: sequential teacher
replay, an attention observer, and saved query configuration need a coordinated
design. The option-name report is not its final implementation-authorizing
design commitment.

## Workspace and inspected evidence

- Source: `/root/.codex/worktrees/6b9d/ILVR`, branch `Stage1-v5`.
- Inspected and copied commit: `148b0e8ead14456d02cc595f0cc32a0752ee22e8`.
- Native managed-worktree creation operation:
  `f259ff39-1554-47c2-a34e-376e412e6036`; creation status returned `created`.
- New workspace: `/root/.codex/worktrees/1061/ILVR`, branch `stage1_v5`.
- `venv`, `data`, and `checkpoints` link to their canonical repository paths.
- Canonical AGENTS.md and source-worktree AGENTS.md were read. No training.
- History spec SHA256:
  `e63bd2cb7deb580c7a7f751c4a7ac814fae7a2b7fd95fee64cd5259e9b151ad8`.
- History plan SHA256:
  `f9769dea4cead2b187b5d226c017d2cb28db45066f0c7bd0bada809676c03f97`.

The documents describe commit `13a67ed2ee975d6ca56b4e9eba18d68b352b166b`,
full fine-tuning, and an updating EMA teacher. Current v5 instead requires
DoRA, ZeRO1, four ranks, microbatch1/GAS4, and frozen EMA tau1. The documents'
CE1, launcher examples, and single-node smoke examples do not override the
standing launch policy. Current v5 already registers an optimizer callback;
the old document's EMA diagnosis is not a verified defect in this revision.

Static call trace: `CustomTrainerStage1V5.compute_loss` uses inherited
`_prepare_stage1_loss_inputs`, which calls `Stage1TeacherTargetBuilder.build`.
Its `_build_sample` performs one teacher language pass per sample before
selection. `textual_states=previous` combines user intent, current text mean,
and previous selected raw-feature mean. `all` instead accumulates text tokens
and selected raw features. Neither branch implements attention-weighted past
injected latent hidden states. Candidate selection is delegated to `_select`.

Recommendation to report: retain the spec's `--history_query_mode` with
`legacy` as the default and `adaptive` as opt-in. Associated proposed settings
are discounted-segment text/gamma0.8, attention-mass latent history, and the
last decoder layer. They are proposed settings, not supported CLI or measured
optimal values. This concerns helper supervision-target selection during
Stage1 training. Sequential prefix replay introduces additional teacher work;
the existing one-language-call target test does not prove adaptive cost.

## Lifecycle evidence and next prerequisites

- `using-superpowers`: read and applied before exploration.
- `readchk`: internal interpretation cross-checked against documents and code;
  report-first scope is resolved. No user clarification required for this stage.
- `brainstorming`: read and applied; context exploration complete. Architectural
  requirements, approaches, section confirmations, written spec/self-review,
  critiques, final commitment approval, planning and method selection remain.
- `using-git-worktrees`: read and executed using native tools, explicit source
  commit, separate branch, shared repository environment, and baseline tests.
- `verification-before-completion`: read and executed for the exact workspace
  and baseline-test claims. No product-completion claim is made.
- `hate`/`macrothink`: not invoked; a complete proposal and the design commitment
  gate have not been reached. No agents were spawned.
- No product code/configuration changes, implementation, TDD, or reviews.
  Branch finishing does not apply at this investigation-only phase.

Future prerequisites, in order: resolve current-v5 scope and compatibility;
compare approaches and confirm design sections; write/self-review the spec;
freeze and critique the proposal under the standing Paperthin gate; obtain
approval of the exact commitment; create an implementation plan and obtain
the execution-method selection; implement with applicable TDD and independent
specification/quality review; verify and finish under the selected lifecycle.

## Fresh baseline verification

Command, in the new workspace:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q \
  tests/stage1/test_targets.py tests/stage1/v5/test_config.py \
  tests/stage1/v5/test_resume.py \
  > logs/stage1-v5-history-query-intake/baseline-tests.log 2>&1
```

Exit 0; complete output read: `99 passed in 3.56s`. These existing CPU tests
cover the selected legacy target/config/resume baseline, not adaptive query,
real-model attention, DoRA compatibility of the proposed adapter, four-rank
training, memory, throughput, or performance improvement.

`git diff --stat` was empty and `git status --short` was empty before this
controller record was added. The record is the only authored source-tree
file in this intake. Test logs/results are repository-local metadata only.

## Follow-up: gamma and architectural section confirmations

User instruction: `history_text_gamma는 0.9로 하자. 그 다음 단계를 진행해라`.
This supersedes the initial recommendation and the supplied document's gamma0.8
for the new feature. It does not rewrite any historical checkpoint or resume.
No product configuration has been edited yet; 0.9 is the agreed design default.

Presented implementation structures:

1. Add legacy/adaptive dispatch within the existing target builder, with prefix
   replay and reduced-attention calculations separated into small modules.
2. Introduce a separate adaptive builder class, preserving the original builder
   at the cost of an additional class and integration boundary.

User selected `추천 구조로 진행` in response to the explicit architecture
question. Confirmed section: option1; gamma0.9; preserve existing v5 DoRA,
frozen teacher tau1, and loss composition. This is approval of that section,
not final written-spec approval or implementation-method selection.

The next section has been presented for confirmation through an asynchronous
question: discounted segment-text history with gamma0.9; current reasoning
attention to previously injected active latent positions, retaining attention
mass and using the full valid causal context as the softmax denominator;
save query settings/formula identity and reject conflicting adaptive resumes;
verify math, actual injection, causality and legacy preservation. Its answer
is pending as of this record update. Dependent spec writing has not started.

Additional static evidence at the unchanged source commit:

- `resolve_qwen2_5_vl_model` unwraps engine/PEFT wrappers via `get_base_model`;
  the proposed adapter must retain the effective wrapped projection modules,
  rather than constructing a plain model that loses DoRA effects.
- The native SDPA attention path calls q_proj/k_proj, applies MRoPE, expands
  GQA keys, and applies the causal mask. Projection hooks can observe those
  module calls; reduced-probability accuracy remains untested for the new path.
- Decoder layer execution passes shared position embeddings. The new observer
  must consume those exact embeddings and actual effective projection outputs.
- Existing `latent_hidden_states` injection validates `[N,H]` against a boolean
  `[B,T]` mask and uses masked_scatter. The new builder must preserve that
  target ordering and output contract, while prefix replay injects only targets
  selected at earlier blocks.
- Existing `Stage1EmaTeacher.update` returns immediately at decay1. The new
  feature must not replace this with the document's updating-EMA algorithm.
- A static integration map includes `src/stage1/targets.py`, new history math
  and observer/prefix helpers, `src/utils_deepseed.py`, `src/main.py`, current
  `src/multinode.py`, and v5 contract/save/resume wiring. Exact edits belong
  in the later approved implementation plan.

These are source observations, not new runtime or performance evidence. No
product changes, agents, GPU work, or training have occurred in this follow-up.

## Current intake recheck: option-name report

The currently visible user request asks for the option-name report first.
This read-only recheck does not advance the prior design confirmations or
grant any new design, experiment, implementation, or launch approval.

Canonical AGENTS.md, the worktree AGENTS.md, both canonical History Query
documents, and this existing record were read. Both document hashes above
remain unchanged. The existing `stage1_v5` workspace is reused; no second
worktree was created. Its HEAD remains
`148b0e8ead14456d02cc595f0cc32a0752ee22e8`.

The source trace was rechecked through v5 `compute_loss`, inherited
`_prepare_stage1_loss_inputs`, and `Stage1TeacherTargetBuilder._build_sample`.
The CLI currently exposes `textual_states=previous|all`; the proposed
`history_query_mode=legacy|adaptive` is not yet implemented. Recommend the
document's name because adaptive changes teacher context and latent-history
weighting as well as text pooling. Actual implementation remains architectural
brainstorming. This report is not the final design-commitment gate, so hate
and macrothink are not invoked here.

Verification scope for this report: branch/HEAD, linked-worktree identity,
canonical document hashes and naming contract, current parser/target source,
and empty tracked diff. No tests are rerun for this read-only investigation;
no adaptive correctness, performance, or training claim is made. Branch
finishing does not apply. The existing workflow record is the only changed
source-tree file.

## Follow-up: user-selected CLI name and pending data-flow confirmation

User instruction: `option이름은 --query_mode로 하자. 계속 진행해`.
This approves the public option spelling and continuation of brainstorming.
It does not approve a not-yet-written spec, plan, execution method, or training.
The source CLI is unchanged; historical recommendations above are preserved
as history rather than represented as implemented settings.

The existing gamma0.9 and architecture-section records were reconciled before
continuing. The remaining data-flow/validation section has been presented via
`request_user_input_async`: legacy/adaptive dispatch, sequential injected
teacher prefixes, discounted-segment text with gamma0.9, full-causal-context
attention probabilities and preserved latent attention mass, existing v5
DoRA/frozen-teacher/loss and candidate selection, saved query semantics and
resume conflicts, plus math/injection/causality/attention/legacy/resume checks.
The question asks approval to write the current-v5 design spec. Its response
is pending; no dependent spec writing or implementation has begun.

Read-only source checks at unchanged HEAD:

- `Stage1EmaTeacher.__init__` owns a frozen eval replica and disables gradients;
  `update` returns immediately for decay1.
- Native SDPA forward calls the effective q_proj/k_proj modules before MRoPE,
  GQA expansion and causal attention. Actual observer accuracy still needs
  tests after approved implementation; source availability does not prove it.
- The existing target result uses packed `[N,H]` values and a `[B,T]` mask.
  The adaptive design must retain the current repository contract instead of
  blindly copying the older supplied spec's `[1,N,H]` return layout.
- v5 resume validates explicit inputs against its saved execution/objective
  contract. Query fields need deliberate integration into this existing path.

Commands: `git diff --numstat`, `git status --short`, `git rev-parse HEAD`,
and `rg -n -- '--query_mode' src docs/superpowers/specs tests/stage1/v5
run_multinode_stage1_training.sh`. Empty tracked diff, HEAD
`148b0e8ead14456d02cc595f0cc32a0752ee22e8`, only this untracked workflow record,
and no existing query_mode CLI match. No tests, GPU work, training, agents,
hate/macrothink commitment passes, or branch finishing in this continuation.

## Approval to write the spec and frozen proposal v1

The user replied `ㅇㅋ` to the immediately preceding request to write a
current-v5 spec for the presented data-flow/validation section. This completes
that section confirmation and authorizes the Superpowers design artifact.
It does not approve the written spec or an implementation method.

Root authored and read back the complete
`specs/2026-10-15-stage1-v5-history-query-design-proposal-v1.md`, checked it
for placeholders, contradictions, scope and ambiguity, and froze proposal v1.
The reference documents' EMA/full-tuning/ZeRO3/shape/launcher assumptions are
explicitly reconciled with current v5. Extra ablation machinery is excluded.
No product code or settings were changed.

Root invoked hate at the design commitment gate. Its exact two-key result is
`specs/2026-10-15-stage1-v5-history-query-hate-v1.json`. The proposed first nail
requires a new tiny adapter/observer and is implementation-like, so it has not
been run and has no experiment authorization. Numerical native equivalence,
real-length cost/VRAM and performance remain unverified. The final commitment
must surface that uncertainty rather than claim the first nail passed.

Macrothink applies: getting causal teacher/DoRA/attention/resume semantics
wrong would waste substantially more implementation effort than three bounded
fresh reads. Root will classify all reads against the same frozen proposal v1.
Fresh readers receive only the stripped factual packet
`specs/2026-10-15-stage1-v5-history-query-macrothink-input-v1.md`, with no
proposed implementation, naming recommendation or other readers' conclusions.
This is the standing AGENTS-authorized, read-only macrothink fan-out.

## Critique outcomes and design commitment v1

Frozen proposal SHA256:
`1e334983ab9084bbb3212828eb9caae48e6d81cc12fd5c5bb046b4b640cc0bc9`.
Root performed hate and collected all three bounded fresh-reader final answers:
`/root/history_query_fresh_a`, `/root/history_query_fresh_b`, and
`/root/history_query_fresh_c`. All were isolated `fork_turns=none` spawns with
the inherited root model, no model override, and the same stripped packet.
No implementer/fixer/reviewer agents were dispatched.

`specs/2026-10-15-stage1-v5-history-query-critique-v1.md` records tasks,
independent returned results, classification and controller disposition.
Incompatible divergence comes first: the smaller one-pass pooling alternative
does not provide target-conditioned teacher reasoning. The user-confirmed
injected-prefix/contextual-attention requirement is why root keeps proposal
v1's design, not evidence that it is the better scientific method. The
compatible contribution is a two-block substitution witness distinguishing
direct query dependence from teacher-state dependence. Convergence is only
reassurance about existing source observations, not runtime validation.

Root authored `specs/2026-10-15-stage1-v5-history-query-design.md` as design
commitment v1, preserving the proposal body and identifying the added causal
witness/empty-text clarification. It explicitly reports the unrun first nail
and numerical/cost/VRAM/performance uncertainty. Final human written-spec
approval and acceptance of that uncertainty remain pending. Only then may
writing-plans begin; implementation-method selection is also still absent.

Branch finishing does not apply to this design-only gate. No product
implementation, premise experiment, training, runtime tests or feature pass
claim has occurred. Verification for this phase is artifact/JSON integrity,
unchanged frozen proposal, source-diff scope and the resulting design commit,
not a project test-suite claim.

Fresh artifact self-review used the repository venv Python with standard-library
hashlib/json/re/difflib. Exit0: frozen proposal SHA unchanged, hate JSON has
exactly root/first_nail, Markdown fences balance, no TODO/TBD/FIXME markers,
and final spec contains the approved option/gamma and pending-approval state.
The full proposal-to-commitment diff was read; only the identified clarification,
status and critique synthesis changed. Exact Git/content verification commands,
their outputs, artifact hashes and design commit identity are persisted under
`results/stage1-v5-history-query-design/verification.json` at this gate.

## Final spec approval and planning authorization

The asynchronous reply explicitly selected `불확실성 수용·명세 승인` for the
commitment-v1 written-spec question. The question named the unrun CPU first
nail and unverified native numerical equivalence, real-length cost/VRAM and
performance. This is the human's explicit uncertainty acceptance and final
written-spec approval for planning, not evidence those premises passed.

Approval binds to `specs/2026-10-15-stage1-v5-history-query-design.md` at
design commit `9c402ff2daa49c1e293b4eb0829f3402389f2839`, SHA256
`9ad3acbf64a4fc17ea25332591dddbb351a47c3dc02f6bf2ecdac0f7e69aae40`.
The approved bytes, including their creation-time pending-approval heading,
are retained; this ledger records the later approval state.

Brainstorming completed its architectural path and root invoked/read
`superpowers:writing-plans` as the sole next skill. Root authors and self-reviews
the implementation plan; no planning agent is needed or dispatched. This
task still has no chosen execution method. No implementation skill, feature
code, new tests/probes, training or evaluation is authorized/executed yet.

Planning source discovery located the actual shell launcher and
`src/multinode_runtime.py::validate_stage1_arguments`; hypothetical Python
launcher-module names do not exist and are not plan targets. The shell worker
always appends `--textual_states previous` for fresh runs, and the current v5
recipe hardcodes it. The plan must distinguish that generated fallback from
an explicit caller request so adaptive can launch and resumes retain omissions.
Current tiny Qwen/DoRA fixtures can support native-reference CPU tests without
downloaded models. These remain observations and planned tests, not results.

## Implementation plan written and self-reviewed

Plan: `plans/2026-10-15-stage1-v5-history-query.md`, SHA256
`d8d9bae862cccaa98f2ee9d383a4a1f3ec1fa6c2704cd6072fb7992084d0cc3c`.
Root completed writing-plans' self-review against the approved spec: six
reviewable tasks, exact interfaces/file paths, 30 future steps with scoped
RED/GREEN commands, all five review-focus classes assigned to tests, and
explicit independent-review/final-verification/runtime-authorization gates.
Self-review clarified tensor shapes, original builder signature/return,
CLI action-destination tracking, generated fallback and saved query omission.

Native is recommended because the six tasks share tightly coupled interfaces;
SDD remains an offered choice. Neither is selected and neither implementation
skill has been invoked. Human plan review and method selection are pending.
No additional agent was used to write/review the plan; the root self-review
is the writing-plans-required check, not independent code review.

Fresh artifact-only verification via repository venv Python: exit0, approved
spec SHA unchanged, six task headings/30 unchecked steps, six interface blocks,
required header/constraints/review sections, balanced fences and no placeholder
markers. No planned pytest command, native comparison, training or other
feature experiment has run. Git/content evidence and the plan commit identity
are recorded in `results/stage1-v5-history-query-design/plan-verification.json`.
