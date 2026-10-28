# Stage1-v5 cross-helper reverse-top-k V-off — design commitment v1

Status: self-reviewed design commitment v1; fresh builder check returned “nothing important left”; pending final human approval. Implementation is not authorized.
Baseline inspected: adbe45492adf2dce6f9fe8d6eee5b0531645c3b2 in the existing externally managed f187 linked worktree. This is a Stage1-v5 feature, not a Stage2 intervention change.

## Intent, agreed scope and constraints

Add optional rejected visual-latent replacement beside existing attention-masked V-off. The user requested same-sample other-step helper donors, both independent and cumulative prefix semantics, original-input-image fallback when no other helper exists, exclusion of the current block's helper, and the same existing clean/teacher selection query in both variants. On 2026-10-24 the user selected prefix-KV reuse with recomputation beginning at the changed latent, after reviewing the alternative of full-sequence rejected forwards.

Success means actual donor features affect the rejected decoder trajectory with correct gradients and causal history. The optional modes must not change chosen/reference DIL, the clean entropy token selector, top-k vocabulary support, entropy/mass formula or weights, Text-off semantics, target positions, normalization, trainer backward/optimizer ownership, or existing mask mode. Improving benchmark accuracy is an empirical question, not a completion claim. No training, evaluation campaign, checkpoint mutation, merge or push is authorized by this design.

## Approaches

1. **Chosen: reuse unchanged prefix KV, recompute affected suffix.** Independent rows use clean prefix up to the current block's first body token. Cumulative uses clean prefix up to the first body token and one causal rejected suffix containing all replacements. This uses existing replay, source-coordinate dropout, and checkpoint bindings. Cache boundaries and gradient equivalence require direct comparative witnesses.
2. **Considered: full-sequence rejected forwards.** Easy reference semantics, but independent branches repeat the long prompt and previous clean steps. Retain this approach as a tiny-test oracle only, not an alternate production backend.

## Public configuration and saved identity

Proposed builder-level interface: one `Stage1V5Config.voff_mode` with values `attention_mask` (default), `cross_helper_reverse_topk_independent`, and `cross_helper_reverse_topk_cumulative`. Expose `--stage1_v5_voff_mode` in Python and `--stage1-v5-voff-mode` through the existing Stage1 worker/coordinator and train/eval convenience script. Do not reuse `preference_topk_mode`: that selects vocabulary support and has different semantics. Omitted fresh options retain mask behavior. Explicit options must survive any sourced stale environment/settings.

Record the selected intervention, donor identity/scope, fallback, rank order, query provenance, replacement-slot policy and replay semantics in a new versioned objective contract. Keep historical schema 1/2/3 validation exact with their historical config-key sets, including their missing-new-option interpretation as attention masking. Do not serialize newly defaulted fields into a historical contract before authenticating it. Historical schema-1 training remains rejected as today. Preserve old artifacts and strict source/reference/query/schedule guards; inspectability does not imply that a historical source-bound run can resume under changed code. New replacement objectives require fresh runs; mismatching saved modes fail before model loading. Save and inherit the new mode on compatible resumes. Do not add a general legacy execution framework.

## Donor selection and provenance

The following detailed rules are controller calls, to be included explicitly in final design approval:

- Use the same per-block query already computed for positive selection in the configured teacher path, for both existing legacy and adaptive query modes. Do not update this query from rejected history or add a teacher/query forward. Keep the clean adaptive ledger positive-only.
- Reuse the already encoded, sample-owned helper vision features. Apply the same existing helper grouping (`helper_group_L`, currently 256) to each image before pooling. Pool all other-step helpers, including earlier and later steps, and rank the union globally; no random donor-image choice or cross-sample access.
- Carry stable source-image identity from prepared examples through sample metadata. Identity means normalized source path for file inputs; for in-memory inputs use stable per-source object identity within the prepared example. Exclude all occurrences of a helper identity assigned to the current block. Deduplicate repeated eligible image identities, keeping their first source order. This is source identity, not pixel-content matching. Existing data has no repeated helper paths within a sample.
- If no eligible other-step helper remains, pool projected original-image feature tokens from the same sample, grouped per image with the same helper grouping. In legacy mode use the actual prepared image-token embeddings returned by the existing teacher forward; in adaptive mode use the existing prepared prompt embeddings. Require real projected features and image-token/grid ownership agreement; ordinary special-token embeddings are not an acceptable fallback. No extra vision forward or pixel crop is needed.
- Compute cosine similarity in FP32 with the existing cosine epsilon. Sort ascending, stable on ties. Stable pool order is source-image order then within-image grouped feature order. Choose the first K distinct candidate positions. Selection is detached, and selected feature values retain their existing representation and model dtype.
- K equals the number of latent-body slots in the current block (currently 8). Fill the entire rejected body; START/END remain unchanged. This is independent of positive coverage selection count: any unfilled positive placeholder is also replaced in the rejected body. No separate latent-k option is needed.
- If a nonempty donor pool has fewer than K candidates, or fallback is missing/invalid, fail with source/block identity before student/replay execution. Do not silently reuse the current helper, repeat features, leave positive features in rejected slots, or skip a sample. Finite features, query, score and output are required for the new selector. Zero-slot spans are invalid for this new mode.
- Minimal debug evidence identifies block, query provenance, donor source/image/grouped index, fallback status and replaced token positions; existing optional diagnostics also tally helper-donor versus original-fallback blocks so future analyses can distinguish them. These are descriptive counters, not effectiveness metrics; keep them out of product user flows. Disabled mode does not build rejection candidates or add forwards.

Raw TRAIN audit (read-only, not a feature test): 3,450 samples; 924 have one helper; all 924 have one original image. No sample repeats a helper path. This establishes source availability only, not projected feature counts, ranking usefulness, or model behavior.

## Replay and gradient semantics

Let a block have START at s, body at s+1 through e-1, END at e. For a target token at t, score decoder position t-1 using the original teacher-forced text.

**Independent.** One row per block with auxiliary targets. Clean attached prefix KV is sliced strictly before c=s+1. Replay begins at c, injects donor features at every body position, retains END and original text, and ends at the last required target query. Only the current block is replaced. Previous latent blocks and text retain clean history. Do not reuse clean KV for any replaced body position.

**Cumulative.** One row per sample with auxiliary targets. Reuse attached clean KV only before the first original latent span's first body token (including an empty-target first span), then execute a causal suffix through the last required target query with every encountered latent body replaced. This includes blocks with no immediate eligible targets when they precede later targets. Earlier replacements affect later states; future replacements cannot affect earlier logits. One causal teacher-forced suffix is equivalent in specified semantics to stepwise continuation with retained rejected KV; it requires no autoregressive text generation or mutable cache.

Both modes keep normal causal visibility of the replacement body. The existing latent-body attention block must not also mask these features. Preserve original position IDs (including multimodal RoPE), padding, boundaries, target/query ordering, vocabulary projection chunking and immediate-text scope.

Copy or functionally scatter into clean prepared embeddings without mutating the trace. Overwritten latent embeddings are detached donor features; gradients must not leak through overwritten clean helper values. Reused clean prefix KV stays attached so rejected losses backpropagate through the unchanged prefix. Recomputed suffix and original later text remain attached to their native trainable paths. Never detach the whole replay prefix to save memory.

Use existing `V5Execution` source-coordinate dropout, so matching original token coordinates receive the same masks across clean, replay and checkpoint recomputation. Keep one native optimizer-owned backward across all terms, with non-reentrant checkpoint recomputation. Separate decoder calls may share the existing single `execution.graph('packed')` scope; each retains its own checkpoint closure and packed-coordinate context.

## Text-off, execution and expected call structure

- Existing attention_mask mode retains its current combined V-off/Text-off packed invocation.
- Independent replacement packs per-block replacement rows with the existing independent clean-prefix Text-off rows in one decoder invocation, allowing row-specific prefix cutoffs, suffix starts and masks.
- Cumulative replacement uses one causal rejected suffix decoder invocation. If Text-off is active, use the existing separate packed Text-off-only invocation under the same overall packed graph scope. This avoids padding every Text-off row to the entire cumulative trajectory length.
- Text-off always uses clean positive latent history and its existing independently selected previous-text masking; replacement history never leaks into it. Helpers, clean student and frozen reference remain unchanged in count for a given configured teacher query mode. Do not promise unchanged total decoder count: cumulative with Text-off has two OFF decoder invocations; independent has one combined OFF invocation. Backward recomputation is additional and is not counted as an original forward.
- The existing need_voff/need_textoff behavior and empty-query handling remain. Cumulative replacement uses all prior blocks, not merely the filtered active-target block list. No extra distributed forward collectives; preserve existing ZeRO-1, native parent dispatch and rank-local OFF execution.

## Components and build boundaries

Expected touched paths: `src/stage1/types.py`, `data.py`, `targets.py` (or a small focused donor selector module), `v5_config.py`, `v5_resume.py`, `v5_trainer.py`; optional narrowly required execution/trace changes; `transformers/.../stage1_v5_replay.py`; `src/utils_deepseed.py`, `src/multinode_runtime.py`, `run_stage1_training.sh`, `scripts/stage1_v5_train_eval.sh`; focused tests and usage documentation. Inspect actual launch propagation rather than assuming the shell flag reaches workers.

The builder may settle dataclass layout, helper factoring and exact test fixture placement. They may not change donor scope, query provenance, K/underfill behavior, replay semantics, resume restrictions or loss behavior without a design revision. Keep the patch focused; no caching service, retry/recovery framework, donor bank or new backend.

## Acceptance witnesses and uncertainty

Use TDD and an independent specification-and-quality reviewer. Required focused witnesses:

1. Hand-computed FP32 donor ranking, stable ties, full-current-image exclusion, union-global selection, same-sample isolation, original fallback, duplicate identities, K=8, underfilled-pool rejection and all invalid-feature cases. Teacher original-image extraction must prove real projected values, not a token-embedding fallback. Exercise both legacy and adaptive queries with instrumented teacher calls and unchanged positive selections/ledger.
2. A tiny actual Qwen causal decoder comparison against separate full-sequence reference forwards: selected logits, total loss and trainable decoder/DoRA gradients for both variants. Include multimodal position IDs, varying block/target lengths, empty immediate-text blocks, one-block equivalence, two-block divergence and clean-prefix-gradient checks. The reference must build its own full sequence and not call the production prefix-packing helpers.
3. Dropout and non-reentrant checkpointing on/off comparative witnesses using original source coordinates; cumulative V-off plus independent Text-off must run both original decoder calls and one successful native backward without context collisions. Perturb first-block donor features: independent later-block logits stay unchanged, cumulative later-block logits change; earlier logits cannot depend on future donor changes.
4. Historical attention-mask behavior and no-option contract inspection; old schema reads, old source guard preservation, new contract roundtrip/inheritance/conflicts before model work, worker and two-node coordinator dry-run propagation, stale-setting precedence. No actual training is needed for a dry-run.
5. Preserve existing clean/reference/Text-off logits and loss formulas; verify entropy and mass consume the new rejected logits with existing clean-top20 selection and mass default of half entropy weight. Run the smallest critical trainer integration witness plus the project-required completion suite once, after review fixes.

Any future effectiveness analysis must distinguish the raw single-helper/original-fallback population from other-helper interventions rather than treating all replacements as cross-helper negatives. Both donor paths are detached teacher features. No accuracy benefit, adverse-effect sign, latency or memory improvement is asserted.

The main unverified technical premise is correct equivalence of attached-KV replay and full-sequence objectives/gradients under dropout/checkpointing. Comparative tests are required before completion; these have not yet run and are not authorized merely by mentioning them here. Query-low similarity is not established as semantic irrelevance or an effective training negative. Feature/runtime timing, GPU peak memory, real multi-node execution and benchmark effects remain unmeasured; this approval does not authorize a training campaign. No speedup claim is part of the contract.

## Critique synthesis and first-nail status

Frozen proposal v1 is preserved in the adjacent `2026-10-24-stage1-v5-cross-helper-reverse-topk-proposal-v1.md` (SHA256 `f0defadb2e6239d551db947a64ba76d488546534e9d4b2a11802eda9123ecce9`). Root hate and three independent same-model macrothink reads are recorded under `logs/stage1-v5-cross-helper-reverse-topk-design-20261024-050727/`; the input was stripped of preferred implementation framing. No critique saw an executed feature test.

Divergent-incompatible: none surfaced. This does not establish absence of incompatibility. Divergent-compatible: low cosine does not prove adverse evidence, and original fallback is a substantial different population. All three fresh reads are classified divergent-compatible; their cache/gradient observations converge with the existing proposed checks, which is reassurance only. The controller retains the user-requested algorithm and adds explicit donor-source counters and separate future-effect interpretation. It does not substitute a different negative objective.

Root hate: the intervention could appear correct while computing a different objective if clean KV crosses changed latents, prefix gradients detach/duplicate, or dropout/checkpoint contexts differ from the full rejected sequence. First nail: an independently assembled tiny actual-Qwen full-sequence versus replay comparison of logits/loss/parameter gradients, with two blocks, a first block without immediate targets, singleton original fallback, source-coordinate DoRA dropout and non-reentrant checkpointing. Perturb the first donor to distinguish independent from cumulative history and verify future replacements cannot affect earlier outputs. This experiment is proposed, not executed or separately authorized. It becomes a required implementation acceptance test only after authorization for that work.

Changes from frozen proposal: make the cumulative cache cutoff explicitly the first original span even when its targets are empty; record helper versus fallback counts in existing optional diagnostics; clarify separate future effectiveness interpretation and identical detachment of both teacher donor sources. Donor scope, selection, K=8 slots, underfill failure, prefixes, loss behavior and chosen approach remain as proposed. No material incompatible divergence or unresolved reviewer disagreement is concealed.

The final approval request must explicitly present that replay equivalence is still unverified and ask the user to accept deferring this evidence to the required implementation acceptance tests. This is a risk waiver for reversible implementation work only, not evidence of equivalence or permission for training. No costly or state-changing premise experiment is authorized by the approach choice.

## Workflow and approval state

Active phase: architectural/genuinely multi-step brainstorming. Proposal v1 frozen and self-reviewed; root hate and three macrothink fresh reads completed; design commitment v1 synthesized. Fresh builder check `/root/cross_helper_builder_check` returned “nothing important left”; no unanswered builder question or reviewer disagreement remains. Final human approval is pending. The approach choice is evidenced by the user's latest `ㅇㅋ`; it is not approval of this document. Implementation method remains unselected for this new scope. After exact design approval, invoke writing-plans before any implementation skill, and obtain the user's method selection under that workflow. Record approval, reviews, revision identity and claim-specific verification durably. No product code or implementation-like premise probe has been written or executed.

Self-review: this document covers intent, boundaries, both approaches and the chosen one, donor/query provenance, ranking and slot edge cases, cache and gradient ownership, Text-off isolation, expected decoder calls, compatibility, acceptance witnesses and remaining uncertainty. Details explicitly assigned to the builder do not alter the intervention semantics. No claim that tests pass or that the feature is implemented is made.
