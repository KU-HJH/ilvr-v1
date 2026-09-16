# Design proposal v1: same-image evidence replacement

Status: frozen proposal for advisory critique, NOT approved for implementation.
Date: 2026-09-16. Lifecycle owner: root, Superpowers brainstorming, architectural path.

## Understood request and approval boundary

Understood as: extend current ours1 context preference so the comparison branch
reads substituted evidence at the current reasoning step, while scoring the same
teacher-forced target text. Prefer a different earlier reasoning step's latent;
when no earlier eligible step exists, sample the same input image's encoder
features. This is a candidate training method, not a claim of improved grounding.

User-confirmed design sections, in conversation order:

- Same-image scope, including samples with only one reasoning step.
- Image fallback uses vision-encoder output feature tokens, sampled to the latent
  block length (user: "vision encoder가 출력한 feature token에서 기존 latent block과 같은 개수를 sampling하여 교체하는 방식으로 가자.").
- Original-image direct attention remains available in both preference branches
  (user: "유지로 가자").
- Detach the borrowed donor latent copy, so this transplant edge does not send
  gradients back through the donor's generation (user: "그건 끊는게 맞다").

Earlier-only donor selection and the engineering choices below are proposed,
not separately user-approved. Final approval must bind the synthesized written
commitment after critique. No implementation or training is authorized by this
proposal. Root may author required design/workflow control artifacts.

## Source and observed context

Source: /root/workspace/ILVR/.worktrees/ours1, HEAD c460d88 with existing local
changes. HEAD alone is NOT its source identity. Preserve those existing changes
and record the working diff fingerprint in the final evidence record before
creating any implementation checkout. Do not implement against HEAD alone.

Read paths: src/step_preference/{config,sidecar,spans}.py,
src/stage2/{data,trainer}.py, src/utils_deepseed.py,
transformers/src/transformers/models/qwen2_5_vl/{modeling_qwen2_5_vl,
stage2_latent_rollout,stage2_shared_prefix,stage2_v_off_insertion}.py.

Read-only raw TRAIN.jsonl census: 3,450 rows, all one image; raw latent-block
counts {1:924, 2:239, 3:561, 4:1227, 5:286, 6:213}. These are unfiltered rows and
raw plan blocks, not retained training counts or validated scored-step counts.

The model builds image embeddings with self.visual under no_grad, then inserts
them into base_inputs_embeds at image-token positions. Clean latent generation
uses all preceding text/image/latent context. Current context preference scores
clean target IDs; it does not generate negative answers. Existing full-prefix
ON/OFF replay provides a starting execution pattern, including paired dropout
RNG and checkpointing, but its visibility semantics cannot simply be copied.

## Approaches and decision

1. Full-prefix replay per target step (recommended initial implementation):
   reuse the once-computed raw image features and clean latent vectors, construct
   original/replacement inputs, and recompute both decoders from the beginning
   of the prefix through the final predictor row. Simple intervention boundary;
   cost is two additional scoring decoder passes per scored step, beyond the
   ordinary clean CE/latent path.
2. Reuse only pre-intervention prefix K/V: potentially cheaper, but requires new
   replacement-specific cache slicing/position/backward contracts. Defer until
   the plain implementation is validated and a real runtime bottleneck exists.
3. Transplant cached donor K/V: reject for this scope. It imports donor-context
   and position-conditioned states rather than replacing input evidence vectors.

## Exact candidate behavior

Expose opt-in step_preference_prefix_mode=evidence_replacement. Existing modes
and their defaults retain their behavior. The new mode requires enabled Stage
2B context_preference, visual-only, independent branches, original image access
enabled, latent boundaries retained, and lambda_entropy=0. Do not silently reset
an explicit conflicting option; validate it before model loading. Default-derived
boundary suppression becomes false for this new mode. Entropy-on replacement is
outside this first change: there is no established higher-entropy premise for a
substitute that may itself be useful. CE and preference weights, beta, margin,
and the already-supported immediate_text/until_next_latent target scope retain
their configured meanings. No training is launched by this implementation task.

For each validated reasoning step k, with L latent pads:

1. Candidate donors are validated reasoning steps in the SAME sample whose
   latent blocks finish before the current block starts. Uniformly pick one;
   never pick self or future steps. Copy all L vectors from that donor's clean
   rollout, preserving their order, and detach the copied tensor.
2. If no eligible donor exists, take the SAME original image's post-merge
   encoder output vectors already stored in base_inputs_embeds, before latent
   rollout or language-decoder contextualization. Sample L distinct indices
   uniformly without replacement when M >= L. When 0 < M < L, sample L indices
   uniformly with replacement. Sort selected indices so image-token source order
   is retained, including duplicates when needed. No crop/re-encoding,
   normalization, interpolation, learned projector, or other-image source.
3. Freeze selection per (training seed, canonical sample ID, step ID, source
   kind), using a versioned deterministic digest and an isolated CPU generator.
   Do not include rank, global RNG state, optimizer step, or checkpoint call
   count. Store the resulting indices in immutable per-forward branch metadata
   before checkpointed execution. Reuse them on backward recomputation. A sample
   has a fixed donor/index pattern across epochs and resume in this first mode.
4. Original branch uses the clean rollout input vectors. Replacement branch
   changes ONLY current latent-pad input vectors. Keep all input token IDs,
   current-slot mRoPE position IDs, text, earlier latent vectors, original image,
   START/END embeddings and native causal/sliding-window visibility equal. Do not
   copy donor image/latent position IDs. No V-off mask or boundary zeroing applies.
5. Recompute both prefixes to target_text_end - 1 (exclusive input slicing),
   score predictor positions target_text_start - 1 through target_text_end - 2,
   and gather exactly the existing target IDs. In particular, recompute END and
   the first target predictor using substituted evidence. Reuse no clean or
   donor decoder K/V after the intervention. This is final-scoring evidence
   replacement; it does not rerun clean latent generation under a corrupt image.
6. Both decoder branches and original clean latent generation remain trainable.
   Stop-gradient applies only to the transplanted donor copy. Donor latent at its
   ORIGINAL earlier context position may still get gradient through that context
   and its own loss; shared model parameters are not frozen. Image encoder remains
   frozen under the existing contract. Sampled image features have no grad.
7. Use identical dropout draws for the paired scoring passes and the existing
   non-reentrant checkpoint pattern, with per-branch metadata bound before delayed
   backward. Global RNG behavior outside the pair must remain well-defined.

Scores are token-mean log probabilities s_original and s_replacement of the same
target T_k. Preference is mean_k[-logsigmoid(beta*(s_original-s_replacement-margin))].
Total is ordinary clean CE + lambda_preference * preference. No reference model,
generated rejected text, offline negative artifact, or new trainable component.
Zero scored steps give ordinary CE only. Replacement is a synthetic comparison
condition, not an independently certified incorrect evidence annotation.

## Components and integration

- A small pure replacement planner under src/step_preference handles donor IDs,
  deterministic indices, sample/step identity and invariant validation. Preserve
  canonical sample identity from the collator through Trainer to this planner.
- A replacement scorer alongside stage2_shared_prefix consumes clean/base
  embeddings and original positions, builds branch inputs, performs paired replay,
  and returns explicitly named original/replacement scores and source counts.
- Qwen Stage 2 forward dispatches this mode using its existing clean rollout and
  once-computed base embeddings. Existing CE is preserved. Project only target
  predictor hidden rows through lm_head in comparison forwards.
- Trainer supplies seed/identity, dispatches this objective, aggregates original
  and replacement step scores, and reports donor/fallback counts and mean margin.
  Do not label replacement diagnostics as vision-off or as generated rejection.
- Configuration, CLI, worker launcher, multi-node option forwarding and saved
  objective identity carry the new mode and deterministic sampling seed/version.
  Old-mode checkpoint identities retain established compatibility behavior.
- One image per active sample is supported initially. Reject missing/empty image
  features, invalid spans, wrong donor length, multiple images, and incompatible
  modes early where possible. New mode supports native causal/sliding visibility
  under existing eager/SDPA language attention and dense/Flex sidecar selection;
  CPU uses dense, CUDA Flex requires its own execution evidence.
- Preserve the current dirty ours1 source in an isolated implementation checkout;
  follow repository worktree and multi-node policies. No edits to Reference/,
  pretrained weights, datasets, existing experiment outputs, or other worktrees.

## Verification scope

Engineering witnesses (not evidence of better reasoning):

- Known small token/vector fixtures establish exact donor eligibility, matching
  count/order, same-image source, first/single-step fallback, insufficient-image-
  token behavior, deterministic seed behavior and no global-RNG consumption.
- A literal tiny CPU decoder replay independent of the planner/scorer checks
  teacher-forced logits and preference math, same masks/positions, self-substitution
  score identity with matched dropout, and first-target response to an injected
  feature perturbation. Include native sliding visibility.
- Isolated-leaf backward probes distinguish zero gradient on the transplanted
  donor edge from permitted gradients through the original donor prefix position;
  show nonzero decoder/original-branch gradients and preserved ordinary CE.
- Hook the actual Trainer->model path to witness image encoder/latent rollout
  reuse, expected paired scoring calls, no text generation, and no stale K/V.
  Check checkpoint forward/backward use identical sampled evidence and check
  paired dropout behavior against a literal reference.
- CLI/config/launcher and saved identity tests cover this opt-in mode and invalid
  combinations; old-mode regressions are checked against the reviewed baseline.
- Use one focused suite, one critical CPU integration, and required project
  completion checks under the actual workflow. CUDA Flex correctness requires a
  separately reported tiny forward/backward witness; do not infer it from CPU
  passing. No multi-node training or full benchmark is part of design approval.

Scientific premise and proposed cheap first nail:

The load-bearing empirical hypothesis is that preferring original step evidence
over these substitutes provides a useful training signal, rather than learning
to distinguish vector provenance or penalizing useful alternatives. Original
image access remains available by explicit user choice. Raw image features and
generated latents have different provenance. Neither random sampling nor a
different step certifies negative semantic relevance.

Before any expensive training, a possible separately authorized pilot would
blindly assess substitute relevance on a small fixed image/target sample and
compare original-vs-substitute score margins stratified by donor source. Evidence
relevance judgments must be independent of the model's own score and supplied
without condition labels. Margins alone cannot validate this premise; ultimately
improvement needs held-out externally scored outcomes. No such pilot or annotation
has run here. Implementation may be approved only as an experimental capability
with this uncertainty explicit, not as a proven learning improvement.

## Workflow prerequisites and next gate

Read-only investigation is complete for proposal framing. No product code has
been edited and no runtime replacement probe has been implemented or launched.
Method choices above were confirmed in chat; engineering details remain proposed.
After spec self-review, freeze this file and apply root-owned hate; conditional
macrothink (wrong scientific framing is costly) and prism (independent scientific
validity and causal/autograd implementation failure modes). Hooks are read-only.
Any material synthesis change creates proposal v2 and repeats applicable passes.
The final design commitment records actual critique findings, evidence status,
unvalidated premises and explicit risk acceptance requested from the human.
Final written-design approval is still required before writing-plans, delegated
implementation, TDD, independent spec/quality review and fresh verification.
