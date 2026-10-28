# Stage 1 cache — implementation commitment v2

Status: complete written commitment for final review; no cache implementation
has begun. Frozen proposal:
[design v2](2026-10-28-stage1-run-cache-design-v2.md), SHA-256
`1d11731ad490e476efeb54f7a2610257bcbb3b9060736d040a5a9d14ce8fd8b7`.
This commitment adopts that proposal without material alteration.

## User decisions already settled

The latest “그렇게 수행하자. 필수적인 검사가 아니면 학습에서 모두 제외하고”
approves deterministic mode only for cache correctness validation, the existing
fast backward in actual training, and removing nonessential training checks.
Do not request those choices again. The selected implementation method remains
Native with independent specification-and-quality review.

Retain the previously selected experiment-scoped cache: first-epoch original
computation/population, once-only node exchange and coverage completion before
epoch two, subsequent read-only SSD mapping with shared OS pages, same-run resume
reuse/rebuild in preparation, and run-only cleanup after normal scheduled
training/evaluation completion. Fixed helpers cache final targets; EMA helpers
cache frozen visual outputs and recompute current-teacher query/selection.
Eligible frozen V5 reference caches final selected hidden states and executes the
unchanged original lm_head. Dynamic student selection and live alignment noise
remain online. Full logits, original-image visual outputs, cross-run cache
sharing and moving-EMA V5 remain outside this scope.

## Production-check contract

- Start/resume: validate required inputs/config/runtime/checkpoint/cache identity,
  eligibility, topology, worker count, storage and source compatibility once.
- Publication/first epoch boundary: validate record shape/dtype/required geometry,
  integrity, complete writes, full coverage and both-node/worker read readiness.
- Training: perform required lookup/cheap structural checks and fail for missing
  or unreadable records. Preserve semantic reductions, loss/backward/clipping,
  applied-step/EMA ordering and existing framework nonfinite/overflow safeguards.
- Separate verification: tensor equality/hashes, detailed finite scans, gradient
  and RNG dumps, profiler/timing hooks, detailed support/selection statistics,
  duplicate layout audits and diagnostic-only norm recomputation.

Do not add a per-step weight hash, cache equality check, full-tensor scan, or extra
collective merely to monitor the cache. Do not remove a computation that produces
an actual training result because it resembles a diagnostic. Cache-off execution
under the new policy also excludes optional diagnostics. Old saved resume
identities and historical artifacts are preserved.

## Critique synthesis

The independent builder `/root/cache_v2_builder` returned “nothing important
left”. It identified no unanswered planning question; this is not a numerical
or performance pass.

**Hate root:** Exact replay can still differ if skipped frozen calls change RNG,
mutable forward state or the subsequent projection geometry. Deterministic-only
comparison could hide that native-production discrepancy.

**First nail:** For a representative supported sample, compare native live
payloads against lossless reload and the unchanged reference projection, with
RNG/forward-state witnesses; separately compare deterministic live/cache
backward. Stop that layer on the first producer/projection/state mismatch before
full-dataset population. This cache/live witness has not yet been run.

Macrothink used two independent fresh reads of the same stripped v2 constraints;
prior unchanged storage decisions had three v1 reads. The cost of invalidating a
multi-path training experiment exceeds the bounded reads. Neither reader received
the preferred storage/cache-boundary solution.

No divergent-incompatible runtime evidence was supplied. Both reads were
`divergent-compatible`, clustered into:

1. Preserve EMA dependency boundaries and distinguish deterministic validation
   from a production-repeatability guarantee (`/root/cache_v2_fresh_a`,
   `/root/cache_v2_fresh_b`). The proposal already caches only frozen EMA vision
   and explicitly requires online teacher-dependent target evolution.
2. Measure net IO/transfer and first-epoch amortization; cache capacity alone does
   not establish speedup (both readers). The proposal already requires timing in
   the fast path and reporting population/exchange costs. Smaller boundaries
   remain possible future alternatives if evidence rejects the selected layer;
   do not silently widen or replace the approved scope.

Agreement is reassurance, not proof. There is no majority-based correctness
claim and no material unresolved design disagreement. No proposal edits were
made between critique passes.

## Evidence, uncertainty and bounded rollout

Existing V5 uncached A/A matched sampled forward payload/loss/RNG but differed in
gradients and Adam moments. Synthetic SDPA tests repeated under deterministic
mode; this does not prove full-model deterministic repeatability, cross-node
cache replay or updating-EMA cache equivalence. These remain explicit acceptance
gates, as do worker epoch transitions, resume reconstruction and net speedup.

Final approval permits reversible optional implementation and its necessary
bounded verification with these uncertainties understood. It does not assert
those gates have passed or authorize a ten-epoch training launch. Cache-enabled
real experiments stay disabled until required correctness witnesses pass.
Do not invent tolerances to clear a failure. Report a failed premise and revise
that layer before production use. Any materially changed design or costly new
premise experiment outside the approved verification scope returns for review.

The first validation work should be the smallest representative payload/projection
and deterministic A/A/A-B witness; broad coverage comes only after it clears.
Measure throughput using the original fast backward. Preserve the existing
sampler, coefficients, schedule and all old artifacts.

## Next phase and evidence

After final review of this written commitment, invoke `superpowers:writing-plans`,
then implement with the selected Native method, focused TDD, independent review
and fresh claim-specific verification. No new worktree is needed: the active
checkout is already isolated and externally managed. No finish/merge handoff is
claimed at this design stage.

Actual decisions and critique records are in
`logs/stage1-cache-design-20261028/policy-v2-intake.json` and
`logs/stage1-cache-design-20261028/proposal-v2-review.json`.
