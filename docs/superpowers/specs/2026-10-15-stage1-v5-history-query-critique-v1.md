# History query proposal v1: design-gate advisory results

Root controller owns synthesis. These are read-only advisory findings, not
independent code review, implementation approval or runtime validation.

Frozen proposal: `2026-10-15-stage1-v5-history-query-design-proposal-v1.md`.
SHA256: `1e334983ab9084bbb3212828eb9caae48e6d81cc12fd5c5bb046b4b640cc0bc9`.
All classifications use this version; it was not rewritten between passes.

Hate result: `2026-10-15-stage1-v5-history-query-hate-v1.json`, exactly
`{root, first_nail}`. The implementation-like first nail was proposed only.
No premise experiment was authorized or run.

## Macrothink input and independent readers

Root stripped the goal to an optional history-informed Stage1 query while
preserving existing defaults, candidates, model/loss/runtime and checkpoint
contracts. Readers saw current source facts and constraints, without the
proposal, selected algorithm, name or other readers' answers.

Packet: `2026-10-15-stage1-v5-history-query-macrothink-input-v1.md`.
SHA256: `40ea319b059c9aaf6519e6027085f0dd6e742b89ef35d4ee78a0a52ab84ff476`.
Three `fork_turns=none` agents used the inherited root model without overrides.
Their only task was a bounded independent read, with optional allowed source
inspection. Editing, probes, training, Paperthin invocation and child agents
were prohibited. All three returned final answers without requested mutations.

## Divergent-incompatible: scope of the experiment

Shared root: **history-sensitive pooling and target-conditioned teacher
reasoning are different experimental contracts**. The stripped selection goal
alone does not establish a need for sequential teacher context construction.
The readers independently favored a smaller single-pass raw-feature pooling
addition unless actual teacher conditioning is a requirement.

- `/root/history_query_fresh_a`: divergent-incompatible direction relative to
  proposal v1. Recommended recent-text/relevance-weighted selected rows with
  the original teacher pass; sequential injection is justified only by an
  explicit dependency requirement.
- `/root/history_query_fresh_b`: divergent-incompatible direction for the same
  root. Described one-pass pooling and recurrent context construction as
  different experiments and warned that the latter materially expands cost.
- `/root/history_query_fresh_c`: divergent-incompatible direction for the same
  root. Recommended one-pass pooling and deferring recurrent context unless
  visual-history-conditioned reasoning is explicitly required.

Root disposition: retain the contextual algorithm because the user explicitly
confirmed injected teacher prefixes, contextual latent attention and mass
preservation before authorizing the spec. That is a scope decision, not
evidence that this algorithm is necessary or more effective. The smaller
alternative remains scientifically plausible but does not satisfy that
confirmed context contract. No majority vote or consensus inference is used.

## Divergent-compatible: a discriminating observation

Each read proposed the same useful intervention: hold text/current candidates
fixed, substitute earlier selected visual targets, and distinguish changes in
the later query/selection from changes in teacher reasoning states.

Root incorporates this into the final commitment's validation rationale.
Input-embedding capture is the primary injection witness. A synthetic
dependency fixture can require downstream change. An arbitrary real-model
sample need not show a large numerical change; lack of one is not by itself
proof of absent injection. Native prefix/attention comparison remains required.
None of these proposed probes was executed during design.

## Convergent observations — reassurance only

The reads agree with static source observations: selection follows the
teacher pass, existing selected rows enter query summaries, and the old
global-intent projection helper is approximate attention. Root independently
traced those paths before the pass. Agreement is reassurance, not proof of
the new algorithm's correctness, usefulness or runtime feasibility.

## Returned reader results

The concise returned claims below preserve each independent result and its
limitations; they are not rewritten as approval of proposal v1.

### /root/history_query_fresh_a

Direction: explicit recent-reasoning plus relevance-weighted past selected
visual rows; keep the existing teacher pass, ranking, packing and default;
record distinct checkpoint semantics. Strongest mismatch: post-hoc pooling
cannot condition reasoning states on previous selections, whereas injection
and replay introduce a different sequential algorithm and cost.

Static evidence: `_build_sample` runs the language forward before selection;
later selected rows affect summaries, not those existing hidden states. The
old attention helper does not reproduce the complete native attention path.
Untested: pooling may be useful without conditioning; neither benefit nor
need for faithful attention is established. Proposed two-block substitution
separates teacher-state dependence from direct query dependence.

### /root/history_query_fresh_b

Direction: opt-in recent-text plus relevance-weighted stored selected features
as a pooling experiment; require recurrent context only if conditioning is
the intended algorithm. Strongest mismatch: applying an attention formula to
the original hidden states cannot demonstrate reasoning over prior targets;
repeated teacher passes materially expand the query-only experiment.

Static evidence: teacher hidden states precede the selection loop; selected
rows later enter `_reasoning_query`; global-intent attention omits the full
native position/mask/normalization path. Proposed two-block substitution
records query, reasoning states and indices, plus default parity. Untested:
pooling sufficiency and meaningful text/raw-visual comparison space. If
contextualized evidence is essential, this reader's smaller direction would
answer the wrong question.

### /root/history_query_fresh_c

Direction: optional recent-text/relevance-weighted selected rows, existing
single teacher pass/candidate/packing/loss, defer recurrent construction
unless required. Strongest mismatch: visual-history-sensitive selection is
not visual-history-conditioned reasoning.

Static evidence: hidden states are computed before selection; raw-vector means
enter afterward; approximate existing scores do not establish native attention
faithfulness. Proposed fixed multi-block sample with changed earlier selected
features compares query/scores/indices separately from teacher-state changes.
Untested: useful target changes, recent-text relevance and mixed representation
scoring. No accuracy/cost benefit follows from these observations; conditioned
reasoning requires a separately approved context-construction design.
