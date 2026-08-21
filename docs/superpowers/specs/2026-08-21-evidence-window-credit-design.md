# Evidence–Window Credit Learning for ILVR

**Date:** 2026-08-21  
**Status:** Design approved in conversation; written specification pending review  
**Target branch/worktree:** `ours-trial1`

## 1. Summary

This change adds diagnostics and configurable training objectives for cross-evidence credit suppression in ILVR. The core failure mode is sequence-level preference saturation: a strong contribution from one visual evidence span can make a global preference margin large enough that other necessary evidence–reasoning dependencies receive little outer-gradient signal.

The design preserves ILVR's existing trainer, EMA teacher, modified Qwen2.5-VL path, latent token positions, and original loss. It adds a modular evidence–window engine that:

1. extracts native interleaved latent spans and reasoning windows;
2. scores full, all-null, and single-evidence leave-one-out conditions;
3. measures global saturation and evidence–window relations;
4. adds window, evidence–window, relation-distillation, and logit-residual losses behind zero-default coefficients; and
5. supports either an adapter-disabled fixed reference or an explicit frozen-copy reference for ZeRO-2 experiments.

The first implementation uses **zero-latent replacement**, not token deletion or arbitrary attention blocking. Hidden-state residual KD remains a conditional Milestone 7 after logit-residual profiling.

## 2. Repository Findings

The design is based on the current ILVR implementation rather than a generic DPO trainer:

- `src/main.py` creates `Qwen2_5_VLForConditionalGeneration`, freezes the visual encoder, optionally adds LoRA, uses `trl.SFTTrainer` through `CustomTrainerStage1`, and currently hard-codes `configs/config_stage3.json`.
- `src/trainer.py` creates a full EMA copy of the policy, selects sparse helper-image features, injects the selected features as `latent_hidden_states`, computes the original causal CE loss, and adds cosine similarity supervision at latent positions.
- `src/main.py::collate_fn_stage1` converts assistant images into latent delimiters, replaces each delimiter's original image-token expansion with a fixed number of latent pad tokens, and builds `image_out_mask`.
- `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` injects either helper-image features or supplied `latent_hidden_states` into positions identified by the configured latent token ID. It preserves token count and multimodal positions.
- The training model is forced to FlashAttention 2. Its training attention path accepts causal/padding masks, but it does not safely support an arbitrary query–key mask at the configured maximum sequence length.
- The CoMT training data has 3,450 records. It contains 1–6 latent spans per record; 2,526 records have at least three latent spans. The dominant structure is initial assistant text followed by alternating latent and text segments, plus final text.
- The dataset contains no explicit evidence–window relation labels. Each latent record contains only `type` and `helper_image`; text records contain only `type` and `content`.
- The current `main` branch has no ILVR-specific top-level test suite. The vendored Transformers tree has its upstream tests. A separate local feature branch contains dataset asset tests and an image-root fix, but those unrelated changes are not part of this design.
- The repository environment is Python 3.11, PyTorch 2.7.1/CUDA 12.8, FlashAttention 2.8.3, Transformers 4.50.0.dev0, TRL 0.15.2, and PEFT 0.17.1.

## 3. Goals and Non-goals

### Goals

- Make global preference saturation observable before claiming the research hypothesis is verified.
- Remove inherited global margin from local reasoning-window objectives.
- Preserve all unrelated visual spans, all text tokens, sequence length, and positional structure during LOO evaluation.
- Compute one counterfactual forward per selected evidence span, not per evidence–window pair.
- Keep the DPO reference fixed and separate from the EMA teacher.
- Distill teacher/student evidence–window relations without maximizing evidence entropy.
- Preserve the original ILVR path when all new features are disabled.

### Non-goals

- No token deletion, full visual-context removal as the main method, Shapley attribution, subset enumeration, or pair/group ablations.
- No claim that every reasoning window should depend on every earlier visual span.
- No claim that LOO is the total causal effect of removing an evidence and regenerating the trajectory.
- No attention-weight attribution objective.
- No entropy-maximization regularizer.
- No replacement of `CustomTrainerStage1` with Hugging Face `DPOTrainer`.
- No unrelated data-root, launcher, or vendored Transformers refactoring.

## 4. ILVR-native Layout

### 4.1 Token ID source

Latent start, latent end, latent pad, padding, and tokenizer special-token IDs are resolved once from the tokenizer/config and passed into the layout extractor. The evidence-window package must not hard-code Qwen token IDs.

### 4.2 Visual evidence span

For example `b`, visual span `V_j` is the interior token range between one matched latent-start/latent-end pair. The delimiters are not evidence tokens.

The active evidence positions are the positions inside that range for which the **post-selection** `image_out_mask[b, t]` passed to the model is true. In the current trainer this is `firstK_mask`, not the collator's initial all-latent-template mask. These are the positions where the EMA-selected helper-image vectors are actually injected. Inactive latent pad positions remain part of the fixed sequence template but are not counted as active visual evidence. Layout boundaries may be extracted before teacher selection, but evidence validity is bound only after this post-selection mask is available.

A matched latent range with no active positions remains visible in layout diagnostics but is invalid for a counterfactual pair.

### 4.3 Reasoning window

The extractor uses supervision and interleaving boundaries, not punctuation or decoded strings:

- `R_0` is the supervised assistant-text run before the first latent span.
- For each visual span `V_j`, the native downstream window is the supervised text after `V_j` and before the next latent span, or before assistant termination/padding for the last span.
- A reasoning token must have `attention_mask == 1`, `labels != -100`, lie outside all latent spans, and not be a tokenizer special token.
- Adjacent latent spans do not create an empty window.

`R_0` has no eligible visual predecessor and is retained to exercise and report that edge case, but it is excluded from visual preference losses.

### 4.4 Batch representation

The pure layout layer supports heterogeneous synthetic batches even though the current end-to-end helper-image path trains with per-device batch size one:

```text
evidence_token_mask [B, J_max, T]
window_token_mask   [B, K_max, T]
evidence_valid      [B, J_max]
window_valid        [B, K_max]
causal_pair_mask    [B, J_max, K_max]
training_pair_mask  [B, J_max, K_max]
```

`causal_pair_mask[j,k]` is true when `V_j` ends before `R_k` begins and both are valid. Every reasoning-window record stores `native_predecessor`, so `training_pair_mask` selects the visual span immediately before that window rather than relying on equal raw tensor indices. This remains correct when `R_0` is present or an empty inter-latent text range is omitted. Logical notation uses `R_k` for the window whose native predecessor is `V_k`; `R_0` is the separate pre-visual window. All causal predecessor pairs remain available for diagnostics and relation KD.

This adjacent training policy is deliberate because the dataset contains no relation labels. It avoids treating every earlier evidence as required by every later reasoning step. A future pair selector may consume explicit dataset links, but no teacher-threshold or heuristic selector is part of the first implementation.

## 5. Component Boundaries

The implementation will follow repository naming conventions while separating these responsibilities under a small `src/evidence_window/` package:

- **config:** typed JSON loading, validation, and zero-default behavior;
- **layout:** token IDs, span/window extraction, validity, and pair masks;
- **interventions:** full, all-null, and `loo(j)` latent conditions;
- **scores:** causal-shift token log probabilities and masked aggregation;
- **losses:** global/window/EW preference, JSD, Huber, and residual losses;
- **metrics:** saturation and evidence-contribution summaries;
- **reference:** adapter-disabled and frozen-copy providers; and
- **trainer integration:** condition orchestration, cache lifetime, loss composition, logging, and EMA updates.

Pure tensor logic must not depend on TRL or a 7B model. Trainer-specific objects must not leak into the scoring and loss APIs.

## 6. Conditions and Forward Orchestration

### 6.1 Conditions

- `full`: inject every active EMA-selected latent vector.
- `all_null`: replace every active latent vector with zero.
- `loo(j)`: replace only the active vectors belonging to `V_j` with zero.

The intervention copies the compact `latent_hidden_states` value tensor and changes the entries that map through the true positions of the post-selection `image_out_mask`. The mapping is the same row-major true-position order consumed by `masked_scatter`; it is validated before use. The intervention does not change input IDs, latent delimiters, attention masks, sequence length, RoPE positions, other latent spans, or text tokens.

This is a fixed-token-history, null-latent conditional intervention. It is not an attention-block intervention and not a regenerated total causal effect.

### 6.2 Reuse and complexity

The existing full policy forward supplies the original ILVR loss and the full-condition auxiliary scores. Additional conditions are executed only when requested:

- global/window experiment: one `all_null` policy forward;
- evidence-specific experiment: one `loo(j)` policy forward per selected evidence;
- reference: the same required condition set under `torch.no_grad()`;
- EMA relation/residual teacher: the same required full/LOO set under `torch.no_grad()` only when a teacher loss is enabled. Policy and EMA condition forwards receive the same EMA-selected latent values and the same zero-latent condition mask so that their relation effects are comparable.

Condition outputs are reduced immediately to token scores and selected residual features. Large model outputs are released as soon as their consumers have been computed.

For one role, the target is `O(1 + J_selected)` full-sequence forwards, never `O(JK)`. Conditions shared by multiple enabled losses are cached by role and condition.

`max_loo_spans_per_example = 0` means all evidence spans. A positive cap selects deterministic, evenly spaced spans along the temporal axis and includes the first and last span when the cap permits. The chosen indices and count are logged. The shipped full-coverage config uses zero; memory-limited example configs may set a cap explicitly.

### 6.3 Existing EMA selection forward

`_teacher_build_latents` currently repeats an identical EMA assistant forward inside the latent-span loop. The implementation may cache that deterministic `eval()` output once and reuse it across span selection. A regression test must prove that the selected latent values and mask match the legacy routine within tolerance. This optimization must not add work to the disabled path.

### 6.4 Dropout

Evidence-window experiment configs should set LoRA dropout to zero so that condition contrasts are not dominated by different dropout masks. Startup validation reports active nonzero dropout modules whenever preference diagnostics or losses are enabled. The first implementation warns rather than silently changing dropout or policy mode.

## 7. Fixed Reference Policies

The EMA teacher is never used as the DPO reference.

### 7.1 Adapter-disabled reference

`reference.mode = "adapter_disabled"` is valid only for a PEFT policy whose non-adapter base parameters are frozen. The provider unwraps the accelerated policy, enters the PEFT `disable_adapter()` context, and executes required conditions under `torch.no_grad()`.

Startup validation rejects this mode if trainable non-adapter parameters could mutate the shared base reference. This mode adds no full base-model copy.

### 7.2 Frozen-copy reference

`reference.mode = "frozen_copy"` snapshots the initialized policy before optimizer updates, calls `eval()`, sets every parameter to `requires_grad=False`, and prepares it as an evaluation model through Accelerate/DeepSpeed. It is excluded from optimizer construction and checkpoint serialization.

This mode supports full fine-tuning and is included in the first implementation for explicit ZeRO-2 experiments. ZeRO-2 does not partition frozen model parameters, so the copy is replicated on every data-parallel rank. Startup logs must report:

- parameter count and dtype-derived parameter bytes;
- detected DeepSpeed ZeRO stage;
- that frozen reference parameters are replicated under ZeRO-2; and
- the simultaneous existence of the replicated EMA teacher.

The default launcher remains backward-compatible with the current ZeRO-3 config. `src/main.py` gains `--deepspeed_config`; frozen-reference examples select `configs/config_stage2.json` explicitly. The program does not silently switch ZeRO stages.

## 8. Token Scores and Preference Losses

For target token position `t`, causal alignment is:

```text
log_p_t(C) = log_softmax(logits[:, t-1, :])[target_token_t]
```

For model `M` and reasoning window `R_k`:

```text
S_k^M(C) = sum_{t in R_k} log_p_t^M(C)
```

Preference margins use sums, not length normalization.

### 8.1 Global visual-condition baseline

```text
m_k^all = [S_k^policy(full) - S_k^policy(all_null)]
          - [S_k^ref(full) - S_k^ref(all_null)]

M_global = sum over windows with at least one predecessor of m_k^all
L_global = -log sigmoid(beta * M_global)
```

This is a diagnostic/ablation baseline. It does not solve visual-context deprivation.

### 8.2 Window visual-condition preference

```text
L_window = mean_k[-log sigmoid(beta * m_k^all)]
```

This removes global aggregation but retains the all-null ablation and is therefore an intermediate experimental variant.

### 8.3 Evidence–window preference

For a valid pair:

```text
m_jk = sum_{t in R_k} (
          [log_p_t^policy(full) - log_p_t^policy(loo(j))]
        - [log_p_t^ref(full)    - log_p_t^ref(loo(j))]
       )

L_EW = sum_{(j,k) in adjacent pairs} w_jk * -log sigmoid(beta * m_jk)
       / sum_{(j,k)} w_jk
```

The first implementation uses `w_jk = 1`. No teacher-derived heuristic weight is introduced. Direct window masking makes this equal to the prefix-cancelled identity `G_j(b_k) - G_j(a_k-1)` and prevents earlier-window margin from entering the current pair.

## 9. Evidence–Window Relations and Metrics

The non-reference relation score is length-normalized:

```text
C_jk^M = [S_k^M(full) - S_k^M(loo(j))] / |R_k|
```

It is computed for all causally valid predecessor pairs selected for LOO. Documentation and logs call it a conditional, fixed-history LOO dependency.

For windows with at least two valid candidates:

```text
q_M(j | k) = softmax_j(C_jk^M / temperature)
L_rel = mean_k JSD(q_teacher(.|k), q_student(.|k))
L_mag = mean_(j,k) Huber(C_jk^student, C_jk^teacher)
```

Distribution and magnitude coefficients are independent. A single-candidate window contributes to magnitude but not JSD.

### 9.1 Saturation diagnostics

The implementation logs the global reference-adjusted margin and its exact outer coefficient:

```text
w_pair = beta * sigmoid(-beta * M_global)
```

It also logs local window margins and selected `m_jk` summaries.

### 9.2 Contribution summaries

For descriptive contribution-mass metrics only:

```text
a_jk = max(C_jk, 0)
p_jk = a_jk / sum_j a_jk
```

- top-1 ratio: `max_j p_jk`;
- normalized entropy: `H(p_.k) / log(number_of_valid_candidates)`; and
- evidence count: number of `C_jk > contribution_threshold`.

Windows with no positive mass or fewer than two candidates are excluded from entropy aggregation, and the valid count is logged. These metrics are not optimized as entropy objectives.

Scalar metric numerators and counts are reduced across ranks before means are reported. Full matrices are disabled by default. When enabled at an interval, each rank writes its own JSONL file containing numeric layout/matrices and an opaque observation identifier, never raw text or image paths.

Local window and adjacent-pair margins are also accumulated by logical ordinal (`window_margin/k` and `ew_margin/k`) so every native window/pair is observable without dumping a full matrix every step. Student and EMA-teacher `C` summaries and contribution summaries use separate metric namespaces. Matrix dumps identify observations with rank, global step, and an opaque local observation counter; they do not require dataset text or path metadata.

## 10. Residual KD

The EMA model is the teacher for residual KD.

### 10.1 Logit residual, Milestone 6

For selected adjacent-pair reasoning target tokens, let `r = t - 1` be the causal predictor position for target `y_t`:

```text
delta_h_jkt^M = h_r^M(full) - h_r^M(loo(j))
delta_z_jkt^M = W_lm_head^M delta_h_jkt^M
```

Because ILVR's Qwen LM head is linear and bias-free, this is the exact full-vocabulary logit residual. The implementation partitions `lm_head.weight` into configurable vocabulary chunks:

- Huber accumulates chunk sums and counts;
- cosine accumulates chunk dot products and norms.

It never materializes a full `[selected_tokens, vocabulary]` residual tensor. Huber and cosine are explicit alternative metrics, not an implicit mixture.

`max_residual_tokens_per_window = 0` selects all reasoning tokens. A positive cap selects deterministic, evenly spaced positions including endpoints when possible and logs the actual count.

### 10.2 Hidden residual, conditional Milestone 7

Hidden residual KD is not implemented alongside Milestone 6 automatically. After logit-residual tests and a smoke profile, the implementation reports incremental memory, runtime, and overlap with relation KD. Hidden residual proceeds only after a separate approval that it has independent experimental value.

When implemented, identical student/EMA dimensions use identity comparison. Different dimensions require an explicit student projection and detached/fixed teacher target. Any learnable projection must be optimizer-visible and gradient-tested.

The literal “subtract the visual residual from the full representation and predict” variant is excluded from the default design and may only appear as a separate future ablation.

## 11. Configuration and Objective

`--ew_config` loads a strict, versioned JSON object into `EvidenceWindowConfig`. Unknown keys, invalid coefficients, unsupported interventions, inconsistent reference modes, and invalid caps fail at startup. Diagnostic switches are independent of loss weights: for example, global saturation metrics may request full/all-null reference scores while `lambda_global` remains zero.

The implemented objective through Milestone 6 is:

```text
L = L_ILVR
  + lambda_global * L_global
  + lambda_window * L_window
  + lambda_EW * L_EW
  + lambda_rel * L_rel
  + lambda_mag * L_mag
  + lambda_logit * L_logit_residual
```

`L_ILVR = ce_weight * L_CE + sim_weight * L_sim`, with the existing CE fallback when teacher latents are unavailable.

Shipped configs:

- `baseline.json`;
- `diagnostic_global.json`;
- `window_preference.json`;
- `loo_window_preference.json`;
- `relation_kd.json`;
- `logit_residual.json`;
- `loo_relation_combined.json`; and
- `frozen_reference_zero2.json`.

All new coefficients default to zero. Multiple losses may be combined only through explicitly nonzero coefficients, and startup logs print the complete active objective.

## 12. Backward Compatibility and Error Handling

When all new coefficients are zero and diagnostics are disabled:

- no reference is constructed;
- no evidence-window layout is computed;
- no all-null or LOO forward is executed;
- no auxiliary metric accumulator is used; and
- the current ILVR loss/output path is returned.

Invalid cases behave as follows:

- unmatched, nested, or overlapping latent delimiters: clear error when new functionality is active;
- `R_0` with no predecessor: retained but skipped by visual losses;
- empty text between adjacent latent spans: no empty window object;
- latent span with no active vector: pair skipped and counted;
- example with no valid pair: original ILVR loss only;
- non-finite score/loss: immediate error with numeric layout summary, without sample text;
- heterogeneous batch layout: validity masks control all reductions.

The current end-to-end helper-image builder remains batch-size-one unless separately changed. The new pure layout/scoring utilities nevertheless support heterogeneous batches as required by their tests.

## 13. Memory and Compute

The existing ILVR process already holds a trainable policy and a full EMA teacher. Approximate BF16 parameter storage for the 7B-class VLM is 15–17 GB per full copy before optimizer states, gradients, or activations.

- Adapter-disabled reference adds no full base copy.
- Frozen-copy reference adds another approximately 15–17 GB per rank under ZeRO-2.
- ZeRO-1 partitions optimizer state only; ZeRO-2 partitions optimizer state and gradients but not model parameters. Neither partitions a frozen reference's parameters.
- Gradient checkpointing and the repository's mixed precision remain active.
- Reference and EMA forwards run under `torch.no_grad()`.
- Full-condition results are reused.
- LOO complexity is one forward per selected evidence, not per pair.
- Residual vocabulary computation is chunked.

The implementation must not silently duplicate another model or switch reference strategy. It must report memory semantics at startup.

## 14. TDD Milestones

Every milestone starts with a failing CPU/synthetic test.

### Milestone 1: layout and diagnostics

- exact visual/reasoning indices for `V1,R1,V2,R2,V3,R3`;
- actual ILVR initial `R0` behavior;
- first/last span, adjacent spans, no predecessor, padding, and heterogeneous batch counts;
- no hard-coded IDs;
- top-1, entropy, threshold count edge cases; and
- cached EMA selection equivalence if the existing repeated forward is optimized.

### Milestone 2: global/window preference

- causal token shift and exact window masking;
- prefix-cancellation identity;
- synthetic `m1 >> 0, m2 ~= 0` gradient saturation;
- substantial independent window gradient for `m2`; and
- no loss contribution from invalid windows.

### Milestone 3: zero-latent LOO

- only selected active vectors change;
- other evidence and text values remain equal;
- IDs, length, masks, and positions remain equal;
- only causally valid pairs are exposed; and
- one evidence condition scores every downstream window.

### Milestone 4: EW-DPO integration

- exact reference-adjusted margin;
- adjacent training pairs versus all-causal diagnostics;
- no inherited earlier-window margin;
- deterministic cap selection; and
- policy gradient/reference no-gradient behavior.

### Milestone 5: relation KD

- teacher/student `C` matrices;
- masked softmax/JSD;
- single-candidate exclusion from JSD;
- separate Huber magnitude loss; and
- no EMA gradient.

### Milestone 6: logit residual

- chunked/full small-vocabulary equivalence;
- independent Huber/cosine variants;
- deterministic token cap;
- student gradient/EMA no-gradient; and
- absence of a materialized full residual matrix.

### Conditional Milestone 7: hidden residual

- only after the approved profiling gate;
- identity/projection semantics;
- projection gradient when used; and
- separate config from logit residual.

## 15. Integration and Final Verification

The new top-level suite lives under `tests/evidence_window/` and uses tiny synthetic models/fixtures so routine CPU tests never load the 7B checkpoint.

Required integration checks:

- baseline one-step optimizer smoke;
- diagnostic global smoke;
- window preference smoke;
- EW-DPO smoke;
- relation JSD and magnitude smokes;
- logit residual smoke;
- explicit combined-config smoke;
- LoRA adapter-disabled reference invariance;
- frozen-copy reference invariance; and
- two-process tiny-model ZeRO-2 frozen-reference smoke.

Reference and EMA parameters are snapshotted before an optimizer step and compared afterward. Policy and any active residual projection must receive gradients; reference and EMA parameters must not.

Relevant vendored Qwen CPU tests and `python -m compileall src tests` are run because the current main branch has no ILVR lint/static-check configuration. A final 7B/B200 smoke is run once when needed and recorded. Equivalent successful evidence is reused; an additional validation expected to exceed five minutes or read more than 10 GB requires explicit approval.

Before any completion claim, the workflow runs `superpowers:verification-before-completion` and reports exact commands and results, files changed, implemented objective, `V_j`/`R_k` definitions, LOO semantics, complexity, configs, limitations, and deviations.

## 16. Known Limitations and Deferred Work

- Zero-latent replacement does not have the same semantics as direct attention blocking.
- Fixed-history LOO includes effects propagated through recomputed hidden representations but does not regenerate the text trajectory.
- Adjacent-pair EW-DPO can miss genuine long-range evidence requirements; all-causal diagnostics expose them, while supervised or validated pair selection is future work.
- Single-evidence LOO does not identify synergy or redundancy among evidence groups.
- A positive LOO score is not proof of necessity, and negative scores are retained in raw relation logs even though positive-part mass is used for descriptive entropy/top-1 metrics.
- Frozen-copy ZeRO-2 mode is intentionally memory-heavy and replicated.
- The existing full EMA copy and batch-size-one helper-image path remain architectural constraints.
- Hidden-state residual KD, attention blocking, group ablation, and Shapley attribution are deferred.

## 17. Approved Design Decisions

The following decisions were explicitly approved during brainstorming:

1. modular evidence-window engine with thin trainer integration;
2. adjacent evidence–window pairs for EW-DPO training and all causal predecessor pairs for diagnostics;
3. adapter-disabled LoRA reference plus an explicit ZeRO-2 frozen-copy reference in the first implementation;
4. zero-latent replacement as the first LOO intervention, with attention blocking deferred;
5. direct window-masked prefix cancellation;
6. separate JSD and magnitude relation losses;
7. exact chunked full-vocabulary logit residual before hidden-state residual; and
8. a profiling and approval gate before hidden-state residual Milestone 7.
