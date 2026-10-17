# Stage1-v5 DIL token mean — design commitment v1

Status: self-reviewed commitment awaiting final human approval.
Implementation is not authorized by the preliminary scope confirmation.

## Agreed problem and scope

The user requested individual DIL tokens receive equal weight, and selected
`전역 token 평균으로 진행 (권장)`. Apply this within one optimizer update over
all four ranks and all prefetched accumulation microbatches. Keep each
token's softplus before averaging. Use the existing `stage1_v5` worktree and
the user's selected Subagent-driven implementation method after approval.

Only Stage1-v5 DIL changes. Keep reference, GT positions/masks, query modes,
CE/alignment/Text-off KL, entropy/mass normalization, coefficients and
schedule. No training, evaluation campaign, merge, push or checkpoint
mutation is part of this approval.

## Exact new behavior

For each eligible DIL target token t, use the existing loss:

`l_t = softplus(-log(p_t(y_t)/mu_t(y_t)))
       + sum_v mu_t(v)*softplus(log(p_t(v)/mu_t(v)))`.

For the optimizer window, DIL contribution is
`lambda_preference * sum_t(l_t) / global_DIL_token_count`.
Do not average original blocks, examples, ranks or microbatch means first.
Padding, ignored labels and excluded crossing/special tokens retain their
existing exclusion. Empty blocks have no DIL tokens. A globally empty DIL
window contributes a graph-connected zero without affecting other terms.

This implementation targets the current `selection=all` profile in both
legacy/adaptive query modes. Combining new token-mean training with another
selector fails explicitly before targets or model loading. This restriction
is a deliberate narrow scope: clean/voff/gap have computable fixed selected
counts, whereas intersection needs a different mechanism. None is silently
normalized by the wrong token count.

## Data flow and normalization

1. Reuse current prefetched metadata. Reduce original-block and eligible DIL
   token counts together over the existing four-rank collective for the
   actual accumulation window. Do not add model passes or retain whole-window
   graphs. Do not replace native Trainer backward/optimizer ownership.
2. Return sums of tokenwise chosen/reference DIL terms. Entropy/mass continue
   returning sums of their original-block token means.
3. Split the loss scales. Before unchanged native GAS division and rank
   averaging, DIL uses `configured_GAS * world_size / global_DIL_token_count`.
   Entropy/mass retain `configured_GAS * world_size / global_original_blocks`.
   Partial tails use their actual token count. Native CE/alignment/Text-off KL
   retain their current configured-GAS reduction.
4. Record DIL sum units, token denominator and its own scale in existing
   diagnostics. Correct sampler/runtime reduction metadata and verification
   consumers to describe the new mixed-denominator objective honestly.

## Saved contracts, resume and errors

New training writes a new schema/reduction identity that explicitly records
token DIL and unchanged block entropy/mass denominators. The tokenwise loss
formula itself stays the same. No new user-facing CLI option is needed.

Historical schema-1 sidecars remain strictly readable for inspection and
existing artifact/reference validation. New implementation training rejects
historical block-mean resume before model loading; it never reinterprets or
rewrites those checkpoints. Historical continuation uses its original reviewed
code revision. New token-mean checkpoints resume only with matching saved
objective and the existing strict execution/source/reference/schedule guards.
Changing block mean to token mean requires a fresh run. Do not relax source
hash agreement or create a generalized legacy execution framework.

Reject unsupported selectors, unknown/malformed reduction contracts and
conflicting resume settings before model work. Zero token count is a valid
zero-DIL window rather than a division error. Preserve existing distributed
error agreement and process ownership mechanisms.

## Tests, review and verification

Use TDD with a hand-computed unequal-block example that fails the old block
mean and verifies token sums/counts. Independently verify chosen/reference
gradients and unchanged entropy/mass/native gradients. Identical eligible
tokens repartitioned into unequal original blocks must keep DIL unchanged.
Cover chunking, ignored/padding positions, empty blocks/ranks/microbatches,
all-zero token windows, unequal four-rank counts and partial accumulation.

Use the existing actual CPU parent-dispatch witness to compare accumulated,
rank-averaged gradients against an independent global token-mean objective.
Test strict old-sidecar reads, early old-resume rejection, new roundtrip and
resume identity, unsupported-selector pre-model rejection and real historical
sidecar inspection. CPU tests do not establish real multi-node ZeRO execution.

Follow the retained SDD implementation/fixer route, independent specification
and quality reviews, applicable final review, fresh verification, and the
applicable finishing gate. Keep durable evidence of actual approvals, agent
work, reviewed commits and exact checks. No training is authorized here.

## Critique synthesis and remaining uncertainty

Frozen proposal v1 and its exact root hate plus three fresh reads are stored
in the adjacent proposal/critique/hate artifacts. No critique disproved the
current all-selector approach. Compatible divergence corrected the rationale:
three optional selectors have fixed cardinalities. They remain out of scope.
Convergence is reassurance, not evidence of runtime correctness.

Changes from proposal: avoid a new configurable block/token mode and legacy
runtime branch. Existing strict source guards make historical training resume
under changed source a separate compatibility issue; reject it clearly while
preserving read-only historical contract support. Add token repartitioning as
an independent invariance witness.

First nail remains proposed, not executed: a CPU native-dispatch gradient
comparison with uneven token counts, empty rank and partial tail while other
term gradients retain their denominators. It will be a required test after
implementation approval. No costly/state-changing premise probe has run.

Reweighting prior measurements over the same 16 samples/37 blocks/1,753 tokens
gives Legacy DIL 0.984719363064 and Adaptive 0.985221240514, versus previous
block means 0.827538855978/0.827918072005. These are prior-data reweightings,
not new implementation verification or performance improvements. Actual
implementation, training gradients on GPUs, multi-node runtime and generation
performance remain unverified for the proposed change.

Self-review: no placeholders; scope, denominators, zero case, selector limits,
old/new contract behavior, critiques, tests and remaining uncertainty are
explicit. Final human approval of this exact commitment is pending.
