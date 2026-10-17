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
