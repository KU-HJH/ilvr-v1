# Stage 1 cache — design commitment v1, pending premise evidence

Status: synthesized proposal for review, **not implementation authorization**.
The frozen proposal is `2026-10-28-stage1-run-cache-design.md`, SHA-256
`a12fe9833b074336fda916a7a64722057383dc50659c56caeffad66961197a2e`.
This document records critique synthesis without rewriting that proposal.

## Agreed problem and scope

Speed up repeated-epoch Stage 1 while preserving bitwise losses, gradients and
updates, sample order, RNG, existing objectives and resume semantics. Use separate
experiment caches, populate in the first epoch, complete both nodes' coverage
before epoch two, then read only. Retain interrupted-run caches and remove that
experiment's caches after normal training/evaluation completion. The latest
user message, “그렇게 하자. 더 논해야할 사항은?”, confirms the cleanup policy;
it does not retroactively authorize an unwritten implementation.

Use static preprocessing and fixed teacher targets where eligible; use frozen
helper vision features with updating EMA. V5 additionally stores the frozen
reference's final selected-position hidden states and executes its original
`lm_head`. Reference eligibility follows the full reference input producer,
not the presence of EMA elsewhere. Moving-EMA V5 support is out of scope.

The controller's recommended storage choice remains SSD-backed read-only mmap
and OS page cache, rather than explicit whole-cache RAM residency. This is a
stated implementation recommendation for the final design, not a claim that
the user separately chose an unmentioned RAM limit or memory reservation.

## Independent builder check

`/root/stage1_cache_builder_check` read the written proposal and returned
“nothing important left” under the brainstorming builder-check prompt.
This means no material unanswered planning question was reported. It is not
an implementation review or a numerical/performance verification.

## Hate result

**root:** Frozen/eval weights and lossless serialization do not prove that the
unchanged computation is bitwise stable across times, ranks, forward state and
the full update path. If baseline recomputation itself varies, preserving one
first-epoch realization cannot satisfy the claimed equivalence.

**first_nail:** Replay a short uncached optimizer window twice from the same
model/optimizer/RNG and sample state using the real coordinator, topology and
backends. Compare frozen payloads and a complete optimizer update. An A/A
disagreement falsifies that profile's strict premise before cache implementation.

Status: proposed only; no GPU probe has been run or authorized. Existing source
inspection and CPU payload counting do not establish this premise. No epsilon
relaxation, backend change or human uncertainty waiver has been granted.

## Macrothink synthesis

Three independent fresh reads received the same stripped goal, constraints and
known source facts without the conversation or preferred storage solution.
The pass was applicable because invalidating experiment results or implementing
the wrong training path costs substantially more than the bounded fresh reads.

No reader supplied evidence of an incompatible runtime result. All challenged
the unmeasured bitwise premise; that remains uncertainty rather than a proven
failure. Do not treat their common concern or agreement as runtime evidence.

| Read | Classification | Material addition or alternative |
|---|---|---|
| `/root/stage1_cache_fresh_a` | divergent-compatible | Explicit live alignment RNG boundary; original-image vision outputs are another possible cache target |
| `/root/stage1_cache_fresh_b` | divergent-compatible | Direct selected-position reference logits are an alternative if measured storage/IO justifies them |
| `/root/stage1_cache_fresh_c` | divergent-compatible | Original-image merged features could reduce repeated visual work and storage; preserve exact visual-call grouping |

The compatible findings cluster around exact computation boundaries and transfer
cost. Incorporate the alignment-noise clarification: `trainer.py` forms alignment
targets from the current input embeddings and adds `0.01 * torch.randn_like`
inside `_alignment_loss`. That operation stays online at the original point;
no perturbed target is cached and no random draw is removed.

Do not expand this project to original-image vision caching. It touches the
student's visual/trace integration beyond the selected helper optimization.
Do not replace hidden caching with full-logit caching: the measured payload
estimate is 1.62 GiB versus 120.26 GiB per node at BF16. The proposed unchanged
`lm_head` recomputation still requires bitwise A/B evidence; a smaller payload
does not prove correctness or speed. These alternatives remain deferred, not
settled by a majority vote.

The reads also converge on rebuilding occurrence metadata, preserving dynamic
student selection, completing both nodes' sample coverage and measuring total
amortized time. This is reassurance only. None performed GPU execution.

## Changes from the frozen proposal

No selected scope or numerical acceptance criterion changed. Make the live
alignment-noise boundary explicit in implementation/review. Continue with the
stated SSD/mmap recommendation; format/grouping remain builder discretion.
Do not add generic eviction, recovery, archival or cross-run reuse systems.

## Concrete next evidence request

Before final implementation commitment, request a separate bounded GPU premise
probe. Proposed bound: for the existing fixed V5 and standard EMA profiles,
execute the same initial accumulation window twice uncached, one optimizer
update per arm (four updates total across both profiles). Use two nodes/four
ranks and the existing coordinator, preserve the full intended LR schedule,
seed, inputs, backends and coefficients, and stop after the bounded window.
Use separate run IDs/output locations; do not alter historical artifacts.
Minimal isolated throwaway instrumentation may collect payload, loss, gradient,
optimizer/EMA and RNG digests and stop at the window boundary. Save probe outputs
under the existing results/logs/checkpoints policies. Reuse existing instrumentation
where possible rather than adding a new runner or recovery framework.

This is a probe proposal, not a claim that an unchanged one-update V5 stop CLI
already exists. Validate the concrete stop/instrumentation route before launch.
No ten-epoch training, evaluation sweep, cache implementation, storage migration
or deletion is part of this probe request. Loading and snapshot costs are not
included in the four-update bound; wall time has not been measured.

An A/A pass only clears the cheapest premise check. After design approval and
Native implementation, require the separate cache/live A/B, epoch transition,
cross-node, EMA update and resume witnesses described in the proposal, plus
independent specification-and-quality review and fresh verification. Include
first-epoch writes and node exchange when reporting net speedup.

## Phase and remaining uncertainty

Brainstorming remains active. Builder check, root hate and bounded macrothink
have completed as advisory design steps. The first-nail evidence and final
human written-design approval are outstanding. No product code was changed.
Do not invoke writing-plans or an implementation skill until the applicable
design gate is satisfied. GPU experiments require separate authorization under
`/root/.codex/AGENTS.md` lines 142–147.
