# Same-image evidence replacement — design commitment v1

Status: ready for final human review; NOT approved for implementation.
Date: 2026-09-16. Controller: root. Active phase: Superpowers brainstorming,
architectural path. This is a workflow control artifact, not an implementation.

## 1. Problem, scope and confirmed choices

Understood as: add an experimental comparison condition to ours1 preference
learning that replaces the current reasoning step's latent evidence, while
scoring the same teacher-forced text and preserving original-image access.

User-confirmed method choices are same-image substitution; sampling vision
encoder feature tokens to the latent count when another eligible step is absent;
direct original-image attention in both branches; and detaching only a borrowed
donor latent copy. The relevant replies were, respectively, the same-image and
single-step request, "vision encoder가 출력한 feature token에서 기존 latent block과 같은 개수를 sampling하여 교체하는 방식으로 가자.",
"유지로 가자", and "그건 끊는게 맞다". These choices do not constitute approval
of the complete implementation commitment below.

Add an opt-in `step_preference_prefix_mode=evidence_replacement`; retain existing
modes/defaults. Initial scope is visual-only independent context preference,
per-device microbatch size one, one original image, retained START/END boundaries, original-image
access enabled, and entropy weight zero. Existing CE/preference weights, beta,
margin and `immediate_text`/`until_next_latent` target selection are honored.
Reject conflicting explicit settings rather than silently overriding them.
When boundary suppression is unspecified, derive false for this new mode.

No training, benchmark campaign, model-weight modification, or efficacy claim is
included. The deliverable is a tested experimental capability. Training, if
separately requested later, remains subject to the canonical ILVR training and
two-node/four-rank policies. Public Stage 2B resume is already rejected and stays
unsupported; deterministic sampling does not add a new resume feature.

## 2. Replacement rule and deterministic selection

For each validated scored step k, let L be its number of latent pads and M the
number of post-merge image feature vectors at original-image token positions.
Validate all scored-step blocks against the configured latent size before donor
selection: every donor must have exactly L vectors. Reject malformed lengths;
never truncate, interpolate or skip a malformed donor to activate fallback.

- Uniformly choose among validated scored steps in the same sample whose latent
  blocks finish before k's block starts. Copy the chosen block's L clean-rollout
  vectors in order; self and future donors are excluded. This earlier-only rule
  is an explicit proposal to avoid importing latents generated after target text.
- If that set is empty (including first/single-step cases), sample L vectors from
  the same image's post-merge vision-encoder output already inserted at image
  token positions in `base_inputs_embeds`, before language contextualization.
  Use uniform sampling without replacement if M >= L; otherwise use replacement
  when 0 < M < L. Sort sampled indices to retain source image-token order. Do not
  normalize/project features, interpolate, crop/re-encode, or use other images.
- Use an isolated CPU generator seeded by a versioned deterministic digest of
  training seed, canonical sample ID, step ID and source kind. Do not consume
  global RNG or incorporate rank, optimizer step, epoch or checkpoint call count.
  Use the existing `canonical_sample_id(row)` contract (explicit nonempty sample
  ID, otherwise SHA-256 of canonical source-row JSON) and the parsed
  `ReasoningStepSpan.step_id`, not loader position or rank. Serialize
  `["ilvr-evidence-replacement-v1", seed, sample_id, step_id, source_kind]` as
  UTF-8 JSON with `ensure_ascii=False` and separators `(",", ":")`. Take the
  first eight SHA-256 digest bytes as a big-endian integer and mask to 63 bits
  for a local `torch.Generator(device="cpu")`. Use source kinds `donor` and
  `image`; donor selection is `randint(N, (1,))`, image selection is
  `randperm(M)[:L]` or, when M < L, `randint(M, (L,))`, then sort image indices.
  These calls use only that generator under the pinned repository environment.
  The donor/index pattern is fixed for repeated appearances of a sample. Bind
  the resulting indices before checkpointed forward and reuse them in backward.
- Reject missing/empty features, wrong donor lengths, malformed spans and
  multiple images for active comparison samples. Zero scored steps use CE only.

The census of the raw source is 3,450 rows with one original image each; 924 rows
have one raw latent block. These are not retained-sample or scored-step counts.

## 3. Forward, attention and gradients

Use the existing full-prefix replay pattern as the initial implementation. Reuse
the once-computed raw image features and clean rollout vectors. For every scored
step, replay original and replacement prefixes independently; the latter changes
only current latent-pad input vectors. Keep current-slot position IDs, text IDs,
other input vectors, original image, START/END embeddings and visibility equal.
Do not copy donor position IDs. Do not apply a V-off mask or zero END.

Replay inputs through the last predictor row (`[:target_text_end - 1]`). Score
logits at predictor positions `target_text_start - 1` through
`target_text_end - 2` against the target IDs in
`input_ids[:, target_text_start:target_text_end]`. Recompute END and the first target prediction from the
substituted inputs. Reuse no clean/donor decoder K/V. This replaces final-scoring
input evidence; it does not regenerate clean latents under a corrupted image.

Dense replay passes a 2-D mask through the model's actual native mask path. Flex
replay uses bounded q/k positions, `k <= q`, and, when `config.sliding_window`
exists, `k > q - config.sliding_window`. This matches this checkout's no-cache
native rule even when `use_sliding_window=False`. Do not reuse `_evidence_mask`,
whose current implementation omits sliding visibility. Preserve existing Flex
CUDA and zero-attention-dropout restrictions. Original-image access is preserved
within the same native visibility rule, not extended beyond it.

Detach only the transplanted donor copy. The donor at its ORIGINAL prefix
position, its own normal objective, and shared decoder parameters remain
trainable. Both scoring decoder branches and original latent generation remain
differentiable. The image encoder stays frozen as in the existing implementation.
No additional latent detachment, prefix detachment or parameter freezing is
introduced. Detach the sampled image-feature copy as well; its encoder source is
already frozen, and the replacement decoder still receives gradients.

Paired scoring uses identical dropout draws and existing non-reentrant
checkpointing with preserved RNG. Bind per-branch metadata before backward.
Snapshot CPU and active-device CUDA RNG before the original branch. Run the
replacement inside a fork restored to that snapshot, then restore post-original
RNG state on leaving the fork. Outside a pair, RNG thus advances as for its
original branch. Do not redraw evidence inside checkpoint recomputation.

Full replay is selected over pre-intervention K/V reuse because the latter needs
a separate cache correctness contract. Donor K/V transplantation is a different
intervention and is excluded. Cost is two additional scoring decoder forwards
per scored step, plus checkpoint backward recomputation. Image encoding and
clean latent rollout are reused. Throughput and peak memory are unmeasured.

## 4. Objective, interfaces and diagnostics

For the same target tokens T_k, compute token-mean log probabilities s_original
and s_replacement. Preference is

`mean_k[-logsigmoid(beta * (s_original - s_replacement - margin))]`.

Total is ordinary clean CE plus configured preference weight times this loss.
Require entropy weight zero in this first mode; no claim establishes higher
entropy for useful or misleading replacement evidence. There is no reference
model, generated rejected answer, offline negative artifact or new trainable
component. These substitutes are synthetic comparisons, not certified negatives.

A small pure planner in `src/step_preference` owns donor/sampling metadata and
validation. A scorer alongside `stage2_shared_prefix` performs paired replay and
selected-row projection. Qwen Stage 2 forward supplies base/clean embeddings and
positions. Trainer preserves sample ID/seed and aggregates replacement scores.
CLI/config, worker launcher, multinode forwarding and saved objective metadata
carry mode, sampling seed and sampling-version identity. Existing old-mode
checkpoint identity/compatibility rules remain intact; do not add resume scope.

Use explicit original/replacement diagnostic names. Report donor/fallback counts
and score-margin means separately by source, as well as the aggregate loss.
Source-specific means are weighted by actual scored-step occurrences: accumulate
margin sums and step counts over the logging window/ranks, then divide. A source
with zero count reports count zero and mean zero. This does not change the
existing per-sample step-mean training loss or distributed loss reduction.
The exact donor/feature indices exist in per-forward metadata for tests and
optional inspection; this does not require a new persistent per-sample archive.

## 5. Required engineering evidence

Implement through the mandatory delegated TDD and independent review workflow,
after final design approval and an approved plan. Evidence must cover:

- Exact source/index/count/position selection on known small tensors; first and
  single steps, multiple steps, insufficient image tokens, no future donors,
  seed repeatability and isolation from global RNG.
- A literal tiny CPU replay independent of planner/scorer construction verifies
  teacher-forced logits, mean-score/loss math, self-substitution score identity,
  first-target influence, and configured target scopes.
- Compare Flex visibility entries directly to the actual native
  `_prepare_4d_causal_attention_mask_with_cache_position` output; compare actual
  native dense forwards with lengths below and above the sliding window. A
  CUDA dense/Flex output-and-gradient comparison is a distinct required witness
  for Flex support, not inferred from CPU passing.
- Isolated-leaf gradient witnesses show that the copied donor edge is detached
  while gradients through its original context position and decoder remain
  permitted. Verify unchanged ordinary CE and paired dropout/checkpoint behavior.
- Observe actual Trainer-to-model calls for one encoder/clean rollout, intended
  replay counts, no text generation/stale K/V and identical forward/backward
  sample indices. Validate CLI, launch propagation, identity metadata and old
  modes against the reviewed source.

Use the repository virtual environment, focused causal tests and critical
integration plus workflow-required completion checks. Do not claim full-size
training feasibility, speed, memory, performance improvement or GPU support
without its corresponding evidence. No replacement runtime has executed yet.

## 6. Frozen proposal, critique and synthesis

Critiqued artifact: `2026-09-16-evidence-replacement-proposal-v1.md`, SHA256
`18d3779966466b59c3f2f617696732807f729c8a3877396d184992c628374f8a`.
It remains unchanged. This commitment clarifies its native visibility rule,
requires source-specific margin diagnostics, distinguishes deterministic
selection from unsupported public resume, and includes relevant untracked files
in source isolation. It does not change the intervention, objective, interfaces
or scope proposed there. These are explicit resolutions of underspecified
engineering details, not a new behavior proposal.

`hate` (root):

- root: synthetic replacements may reward representation provenance or context
  mismatch rather than useful step-specific evidence. A same-image substitute
  may be helpful, and original image access remains available.
- first_nail: before costly training, inspect a small preselected sample spanning
  donor/fallback cases, obtain condition-blinded independent judgments of source
  image/step relevance, and separately compare margins by source. Frequent
  equally-useful substitutes or source-type dominance unrelated to those judgments
  would challenge the presumed ordering. Source annotations are proxies, not a
  direct readout of hidden latent contents; this pilot cannot prove efficacy.
- status: NOT RUN. No new probe implementation/inference or annotation was
  authorized; this reversible implementation is proposed as an experimental
  capability. Any such pilot or training needs its own concrete scope and
  authorization. Empirical usefulness remains unvalidated.

`macrothink`: two context-free same-model reads, `/root/replacement_fresh_a` and
`/root/replacement_fresh_b`, each classified divergent-compatible. Shared root:
this measures latent-replacement sensitivity, not an exclusively visual causal
effect. Both reject the stronger grounding interpretation and suggest separating
donor/fallback provenance. Neither contradicts the restricted experimental
capability; neither result is empirical evidence or a vote. The convergent
recommendation to preserve text/image conditions and detach only the copy is
reassurance only. No incompatible divergence is left hidden.

`prism`: `/root/replacement_causal_lens` accepted the causal/autograd contract;
`/root/replacement_execution_lens` initially returned UNCLEAR because native
sliding visibility and its independent witness were underspecified. The single
resolving question was which exact native rule dense/Flex replay must share.
Section 3 and the native-comparative witnesses above answer it. The execution
reviewer then returned RESOLVED for the design contract, preserving unverified
runtime/memory/throughput status. No outstanding design-contract objection was
reported; this is not an implementation review or runtime pass.

`mandela` (root, advisory): an interpretation as removing all visual evidence or
isolating solely visual contribution would use the wrong null, since original
image access and text-conditioned latent content remain. The commitment limits
interpretation to conditional intervention sensitivity. Model margins and
designer-authored engineering tests do not certify semantic negative validity;
future efficacy requires independent held-out outcome scoring. No empirical
validation has been claimed.

## 7. Workflow record and final approval gate

Read/used: using-superpowers (intake), brainstorming (active), hate (executed),
macrothink (two fresh reads), prism (two lenses and focused resolution), mandela
(validation interpretation audit). Root only invoked advisory hooks; their
agents were read-only and did not invoke skills or delegate. Skills read for
subsequent gates: shower; verification-before-completion was already read earlier
in this conversation. Writing-plans/TDD/implementation/review/branch finishing
have NOT started. `shower` ran as an artifact-only cold read by
`/root/replacement_cold_read`: verdict minor gaps; the overall scope, behavior,
gradient boundary, empirical limits and approval state were understood correctly.
Its gaps concerned donor lengths, M, canonical IDs/digest, RNG sequencing,
diagnostic weighting and microbatch scope. Root clarified those exact details
above; no intervention or objective change resulted. The reader considered the
gaps nonblocking for design approval. This is a comprehension review, not a
runtime or implementation review.

Source at intake: ours1 HEAD `c460d88184b69221a2e5b4a87736d1b3ea7c9643`;
pre-existing tracked binary diff SHA256
`f58f8469d938111814c4efe71b075126aa24bc7580b286289ee1344f37131ff7`.
That fingerprint is NOT a complete source snapshot: existing untracked runtime
tests/source must also be preserved and individually hashed before isolated
implementation. No product code/configuration has been edited in this task.
Only these design control documents are authored by root.

Fresh pre-handoff checks: the frozen proposal's SHA-256 still matches the value
in section 6; `git diff --binary | sha256sum` still equals the intake tracked
diff fingerprint; `git diff --check` returned exit 0. The two new design files
are the only task-authored files to commit. The staged document diff receives a
separate whitespace/content check before the specification-only commit. These
are artifact/source-preservation checks, not replacement runtime tests.

Human approval requested: authorize implementation and engineering verification
of exactly this experimental capability, including earlier-only donors, fixed
sampling, preserved image access, donor-copy detach, full replay cost and the
entropy-zero scope. Explicitly accept that substitute relevance and training
benefit remain unvalidated; this is not approval to launch training or to claim
improved grounding. On approval, record the actual reply and document identity,
then proceed to writing-plans and delegated implementation. Material later
changes return to the design commitment gate.
