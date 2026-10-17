# Stage1-v5 DIL token mean — frozen proposal v1

Status: complete critique proposal, not implementation-authorizing approval.
Source: `c9d6afe1293b404243e7a48d0e31aeaf33cab4e2` in `stage1_v5`.

## Problem and scope

The user requests equal DIL token weighting instead of equal original-block
weighting. The preliminary question was answered `전역 token 평균으로 진행 (권장)`
on 2026-10-17. Subsequent global/sample questions were answered: equal token
weight versus equal example weight. No instruction reversed that choice.

Change only Stage1-v5 DIL reduction. Keep tokenwise softplus, GT masks,
reference, query modes, CE/alignment/Text-off KL, entropy/mass reduction,
coefficients and schedule. No training, evaluation campaign, checkpoint
mutation, merge or push is authorized. Retain the user's SDD method.

## Behavior

For one optimizer accumulation window across all ranks, T is its eligible
DIL token set. Keep the current per-token loss:

`l_t = softplus(-log(p_t(y_t)/mu_t(y_t)))
       + sum_v mu_t(v)*softplus(log(p_t(v)/mu_t(v)))`.

New DIL is `lambda_preference * sum_T(l_t) / len(T)`. Do not average block
or example means first. Empty blocks contribute neither loss nor count.
A globally empty DIL window returns a graph-connected zero. Other terms
retain their respective denominators.

Current production uses `selection=all`. This change supports new token mean
with that selector, so the denominator is known from prefetched metadata
before loss/backward. Other selectors with token mean fail before targets
or model work. Historical block-mean resumes retain selector support.
Generalizing dynamic selected-token counts is outside scope.

## Implementation and compatibility

- Save an internal `dil_reduction` identity: new-run default `token_mean`,
  historical resume `block_mean`. No new CLI is required. Both query modes
  use the same saved reduction mechanism.
- Extend accumulation-window metadata with global eligible-token count.
  Reduce token/block counts together using the existing collective; no
  added model pass, changed parent dispatch or manual backward.
- Objective DIL becomes token sums in token mode. Entropy/mass stay sums
  of block token means.
- Split DIL scaling from entropy/mass. DIL token scale before the unchanged
  parent GAS division/rank averaging is `GAS * world_size / global_tokens`.
  Entropy/mass retain `GAS * world_size / global_blocks`. Partial windows
  use actual tokens; native loss terms retain configured-GAS semantics.
- Version new sidecar schema/reduction semantics. Validate original schema-1
  sidecars against their original exact field/semantics contract and resolve
  missing reduction to `block_mean`. Never relabel old runs as token mean.
  Resaving an old resume may write explicit new-schema `block_mean` while
  retaining its represented objective. Reject malformed/unknown contracts.
- Switching an old run to token mean requires a fresh run. Ordinary resume
  inherits saved settings. Preserve all old checkpoint contents.
- Diagnostics identify DIL sum units and denominator; no unrelated logging
  or orchestration changes.

## Verification

TDD: independent unequal-length token-loss example must distinguish token
mean from block mean, then verify sums/counts/gradients and unchanged other
terms. Cover empty blocks/ranks/microbatches, zero-token window, unequal
four-rank counts, partial accumulation, and chunking. Use existing actual
CPU parent-dispatch witness and compare rank-averaged gradients with an
independent global-token objective. This is not a real ZeRO/multi-node run.

Check original schema-1 read/resume, new roundtrip, invalid contracts and
unsupported selector rejection before model work. Also read the two real
historical sidecars on CPU. Independent spec/quality review and fresh
claim-specific verification remain required. No training probe is authorized.

Existing 16-example measurements have 37 nonempty blocks and 1,753 tokens.
Reweighting saved block values yields Legacy 0.984719363064 and Adaptive
0.985221240514 instead of 0.827538855978 and 0.827918072005. These are prior
measurement reweightings, not new forwards, training, performance gains or
verification of a new implementation.

## Alternatives and commitment prerequisites

Global token mean is the selected equal-token weighting. Per-example token
means retain equal example weight and favor tokens in shorter examples.
The existing equal-block mean does not satisfy the request. A dynamic-selector
prepass or changing all auxiliary terms expands scope and execution cost.

Freeze this version for root hate and conditional macrothink, then synthesize
the exact commitment and self-review. Final human approval is required before
planning/SDD/TDD. Critiques or proposed first-nail experiments are not execution
authorization or evidence. No costly/state-changing premise probe is implied.
