# ILVR Stage-2 Evidence–Window Credit Design

**Date:** 2026-08-22

**Status:** Approved for implementation planning

**Repository:** ILVR (`XD111ds/ILVR`)

**Scope:** Preserve Stage 1 node grounding and add exact evidence-to-window credit learning to an explicit Stage 2 path.

## 1. Summary

ILVR Stage 1 learns context-aware latent visual states. This design leaves that node-grounding objective unchanged and gives Stage 2 a distinct role: learning which earlier latent evidence each later reasoning window actually uses.

The main intervention is an exact, window-local attention block. For an eligible evidence–window edge `(Z_j, R_k)`, only attention from query tokens in `R_k` to key/value tokens in `Z_j` is blocked. Previous reasoning, other visual evidence, sequence length, token positions, causal structure, and padding remain unchanged. The resulting full-versus-LOO likelihood difference is a conditional/direct dependency estimate, not a total causal effect.

The correctness-first implementation retains FlashAttention2 for the ordinary Stage-2 CE path and uses a matched SDPA path for both full and counterfactual auxiliary forwards. The default preference objective is reference-free. A fixed Stage-1-initialized policy executed with Stage-2 self-generated latent semantics may optionally provide precomputed DPO, relation, and residual targets.

## 2. Problem and scientific interpretation

For interleaved reasoning

\[
R_1,Z_1,R_2,Z_2,\ldots,R_K,
\]

let `Z_j` be a generated latent visual-evidence span and `R_k` a later assistant reasoning-text window. A global objective of the form

\[
L_{global}=-\log\sigma\left(\beta\sum_{j,k}C_{j\rightarrow k}\right)
\]

allows one salient edge to saturate the common outer sigmoid. This can suppress the marginal learning signal for other task-required evidence–window edges. The target failure is therefore cross-evidence compensation caused by global-margin saturation.

The remedy must not delete all previous visual evidence for a window. Such an intervention would deprive the window of evidence it may still require and would force it to depend too strongly on possibly incomplete or stale textual history.

This project therefore separates edge-level contributions while preserving all unrelated context. It does not assume that every window should use every visual span, that dense evidence use is preferable, or that contribution entropy should be maximized.

## 3. Repository findings that constrain the design

### 3.1 Current Stage handling

- `src/main.py` selects `CustomTrainerStage1` and `collate_fn_stage1` regardless of `args.stage`.
- `config.stage` reaches the custom Qwen2.5-VL model, but `--load_model_path` is currently unused.
- The shared trainer always constructs an EMA teacher, extracts helper-image targets, injects teacher latents, and combines CE with similarity loss.
- In the model, `config.stage == "stage2"` sequentially replaces latent-token embeddings with preceding hidden states. Thus the forward uses self-generated latent states, while the shared trainer still supplies Stage-1-style helper-image/similarity supervision.
- Existing evidence-window tests and the prior 7B validation exercise Stage 1 rather than a distinct Stage 2 trainer.

The current checkout therefore does not provide the explicit Stage-2 baseline required by this work. The authoritative Stage-2 baseline in this design is Stage-1-checkpoint initialization followed by self-generated latent states and assistant-text CE only. The legacy shared-trainer Stage-2 behavior will be characterized where useful, but is not the behavior to preserve as the new Stage-2 baseline.

### 3.2 Dataset and sequence metadata

- Training rows contain `image_input`, `text_input`, `sequence_plan`, and `original_final_answer`.
- The current preprocessing consumes `sequence_plan` while constructing conversations and does not preserve it through collation.
- The training data has no explicit stable sample identifier. A deterministic source-row index is therefore needed for precomputed reference artifacts.
- Runtime token spans must be derived from the actual tokenized sequence and assistant labels, with `sequence_plan` retained as a consistency check rather than treated as a substitute for token indices.

### 3.3 Attention backends

- Training normally loads Qwen2.5-VL with `attn_implementation="flash_attention_2"`.
- The installed FlashAttention interface supports causal, padding, sliding-window, and related structured modes, but not an arbitrary query-key block such as `R_k × Z_j`.
- The Qwen eager and SDPA paths accept additive/4D masks.
- The Stage-2 latent-generation prefix loop currently assumes a 2D attention mask, so an intervention mask must be a separate argument and must be sliced explicitly for each prefix.
- PyTorch FlexAttention is available, but introducing it immediately would add compilation, dynamic-mask, DeepSpeed, and checkpointing risks.

Native FlashAttention2 cannot implement the required intervention exactly. Token deletion, latent zeroing, or a global latent mask are not semantically equivalent and must not be used as silent fallbacks.

## 4. Considered implementation approaches

### 4.1 Matched SDPA auxiliary replay — selected

Keep the ordinary Stage-2 CE forward on the existing FlashAttention2 path. For evidence-window auxiliary computation, execute one unblocked full forward and pair-specific blocked forwards through a same-weight SDPA attention branch.

Both sides of every contribution use SDPA, preventing FlashAttention-versus-SDPA numerical differences from being interpreted as evidence contribution. This approach is straightforward to test on CPU and is the least invasive exact implementation.

Initial cost is one matched full auxiliary forward plus one forward per selected edge. Dense 4D masks and repeated activations are expensive, but correctness is explicit and provides an oracle for later optimization.

### 4.2 FlexAttention block-mask replay — deferred optimization

FlexAttention could express the same block rule more efficiently and may support block-sparse execution. It is deferred until the SDPA path establishes exact expected behavior because dynamic per-example block masks, compilation, B200 execution, DeepSpeed, and gradient-checkpointing need separate validation.

### 4.3 Layer-local KV replay — deferred research optimization

A layer-local replay could hold prior hidden states and K/V fixed while recomputing only the current window. This most directly encodes frozen-history semantics, but requires invasive cache and hidden-state plumbing across the custom Qwen model and complicates gradients and Stage-2 latent generation. It is not appropriate for the first implementation.

## 5. Stage architecture

### 5.1 Stage 1: node grounding

`CustomTrainerStage1` and `collate_fn_stage1` retain the current behavior:

\[
L_{S1}=\lambda_{CE}L_{CE}+\lambda_{sim}L_{align}.
\]

This includes the EMA teacher, helper-image supervision, existing masks, mixed precision, and selective perceptual modeling. No evidence-window auxiliary is attached to Stage 1. A request to enable Stage-2-only EW options under Stage 1 should fail clearly rather than silently change Stage-1 semantics.

### 5.2 Stage 2: edge grounding

`CustomTrainerStage2` and `collate_fn_stage2` form an explicit path selected by `args.stage`.

Stage 2:

1. requires and loads the approved Stage-1 checkpoint through a functioning checkpoint argument;
2. sets `config.stage = "stage2"`;
3. uses the model's self-generated latent-state path;
4. does not create or update the Stage-1 EMA teacher;
5. does not run helper-image teacher forcing;
6. does not add the Stage-1 similarity objective;
7. computes assistant-text CE and optional Stage-2 auxiliaries.

The Stage-2 preprocessing/collation path still converts latent steps in `sequence_plan` into latent marker spans, but it must not decode or validate the corresponding helper-image files. It processes only the original user/input images required by the task. It must not emit `pixel_values_latent`, `image_grid_thw_latent`, or teacher `latent_hidden_states`. The current Stage-1 `image_out_mask` teacher-injection contract is not reused; any token-level latent mask needed by the Stage-2 parser is produced under a Stage-2-neutral name from `input_ids` and configured marker IDs.

With all new coefficients zero, no auxiliary SDPA or reference forward occurs and Stage 2 reduces to self-generated latent states plus assistant-text CE.

DeepSpeed ZeRO stage names must be distinguished in logs and documentation from ILVR training stages. `configs/config_stage1.json`, `config_stage2.json`, and `config_stage3.json` describe ZeRO-1, ZeRO-2, and ZeRO-3 respectively; they do not select ILVR Stage 1 or Stage 2.

## 6. Span and edge definitions

### 6.1 Latent evidence

`Z_j` is the complete latent segment identified using configured/tokenizer-provided markers:

- `<|latent_start|>`
- zero or more `<|latent_pad|>` positions
- `<|latent_end|>`

Token IDs must come from tokenizer/model configuration utilities. They must not be duplicated as hard-coded constants in the scorer.

### 6.2 Reasoning windows

`R_k` is the assistant text-token span between latent segments. A window mask includes only target-bearing assistant text positions. It excludes:

- padding;
- user and system prompt tokens;
- latent start, pad, and end tokens;
- non-assistant tokens;
- positions excluded by the existing label mask.

The parser operates on tokenized inputs and labels. Preserved `sequence_plan` metadata is used to assert or test that the expected text/latent order agrees with token parsing. Batched examples may contain different numbers of evidence spans and windows.

### 6.3 Eligible edges

The exhaustive edge set is

\[
\mathcal E=\{(j,k): Z_j\text{ occurs causally before }R_k\}.
\]

The initial reasoning window before any latent evidence participates in CE but has no EW edge. Empty windows and windows without a prior evidence span yield no edge. Edge reductions must remain finite for these degenerate cases.

When `max_evidence_edges_per_window` is unset, all valid edges are used. A configured cap selects the most recent preceding evidence spans first, breaking any tie by ascending token start. This deterministic nearest-first cap is an approximation of the exhaustive candidate set; it does not change the intervention meaning of an included edge. Relevance-based pruning requires a separately documented policy and is not part of the first implementation.

## 7. Exact window-local LOO intervention

For a selected `(j,k)`, define an additive block mask at every language self-attention layer satisfying

\[
A_{t,s}=-\infty \quad\text{iff}\quad t\in R_k,\ s\in Z_j,
\]

in addition to the normal causal and padding constraints. All other allowed attention edges remain allowed.

The model receives the normal 2D padding mask and a separate intervention description or mask. The separate mask is combined only in the mask-capable auxiliary attention path. In the Stage-2 prefix loop, it is sliced as `[..., :prefix_length, :prefix_length]`; the normal padding mask continues to drive position/RoPE and existing prefix behavior.

The intervention preserves:

- all token IDs and embeddings outside ordinary Stage-2 evolution;
- sequence length and position IDs;
- all other evidence spans;
- all previous text and latent computation;
- access from `R_k` to history other than `Z_j`;
- normal causal and padding rules.

Because only query rows in `R_k` are modified, `R_<k` is causally unchanged. A pair forward may alter future states, but only `R_k` is scored, so those later changes cannot affect the measured contribution.

Blocking one `Z_j` for several downstream windows in a single forward is not equivalent: it would alter an earlier window that becomes part of the retained history for a later window. The correctness implementation therefore uses pair-specific interventions.

There is no automatic fallback to latent zeroing, token deletion, or global masking. If SDPA cannot be executed in a supported environment, EW training must stop with an actionable error while the ordinary baseline remains available.

## 8. Scoring and losses

### 8.1 Matched full and counterfactual scores

For auxiliary scoring, the policy performs:

1. one unblocked full-condition Stage-2 forward through the SDPA branch;
2. one SDPA forward for every selected edge with the exact pair mask.

For model `M`, the default window score is mean log probability:

\[
\ell_M(R_k\mid H)=\frac{1}{|R_k|}\sum_{t\in R_k}\log p_M(y_t\mid H,y_{<t}).
\]

The optional `sum` reduction uses the unnormalized sum. Token log probabilities follow the repository's causal shift convention, and window masks must be shifted consistently with labels.

The evidence contribution is

\[
C^M_{j,k}=\ell_M(R_k\mid H_k^{full})-\ell_M(R_k\mid H_k^{-j}).
\]

This is a frozen-history conditional/direct re-reference dependency estimate. It is not the total causal effect of removing `Z_j` and regenerating the complete downstream trajectory.

### 8.2 Reference-free preference — default

\[
L_{EW-pref}=-\frac{1}{|\mathcal E|}\sum_{(j,k)\in\mathcal E}w_{j,k}\log\sigma(\beta C^\theta_{j,k}).
\]

Initial edge weights are uniform unless an explicit, documented weighting configuration is selected. No-edge batches return a differentiable or device-correct zero auxiliary and fall back to CE without NaN.

### 8.3 Fixed-reference DPO — optional

When fixed reference contributions are available,

\[
M_{j,k}=C^\theta_{j,k}-\operatorname{sg}(C^{ref}_{j,k}),
\]

\[
L_{EW-DPO}=-\frac{1}{|\mathcal E|}\sum_{(j,k)\in\mathcal E}w_{j,k}\log\sigma(\beta M_{j,k}).
\]

The reference is not the Stage-1 EMA teacher. It is a fixed copy of the approved Stage-1 initialization policy evaluated with `config.stage = "stage2"`, self-generated latent states, no helper-image teacher forcing, and the same matched-SDPA contribution definition.

### 8.4 Global diagnostic

A global aggregate may be logged to diagnose saturation but is not added to the local training loss. Tests compare

\[
L_g=-\log\sigma(\beta(m_1+m_2))
\]

with

\[
L_w=-\log\sigma(\beta m_1)-\log\sigma(\beta m_2).
\]

For `m_1 = 20` and `m_2 = 0`, the gradient of `L_g` with respect to `m_2` must be approximately zero, while the independent-window gradient magnitude remains `beta / 2`.

## 9. Reference execution and artifacts

### 9.1 Modes

- `none`: default; reference-free preference, no reference model or artifact.
- `precomputed`: recommended DPO/relation/residual mode for large-model training.
- `online_frozen`: included for explicitly validated ZeRO-1/2 configurations; forbidden under ZeRO-3 because it requires a second full parameter copy.

No path may silently switch reference semantics to resolve memory pressure.

### 9.2 Precomputation

A dedicated evaluation command computes fixed-reference targets with dropout disabled and gradients off. Artifacts are keyed and validated using at least:

- deterministic dataset source-row index;
- reference checkpoint identity;
- tokenizer and relevant special-token identity;
- sequence/parser version;
- contribution reduction;
- attention intervention/backend semantics;
- edge-selection configuration;
- relevant model configuration.

Artifacts contain scalar `C_ref[j,k]` values and, when residual KD is requested, the pooled hidden residual vectors defined below. A mismatch is a hard error. The producer's validated artifact witness is reused; training must not recursively rehash a large immutable checkpoint.

### 9.3 Online frozen reference

The first implementation includes the online frozen-reference variant for ZeRO-1/2. It instantiates the fixed model from the Stage-1 checkpoint independently of the trainable policy, before policy mutation, and freezes every parameter. It must not alias the EMA teacher or receive optimizer state. Its memory cost is approximately one additional full set of model parameters, plus temporary reference-forward activations; reference forwards run under `torch.no_grad()`. ZeRO-3 rejects this mode and directs users to precomputed or reference-free mode. In particular, it must never `copy.deepcopy()` an already DeepSpeed-managed policy.

## 10. Relational distillation

For each window with eligible evidence, normalize only over its candidate evidence set:

\[
q_M(j\mid k)=\operatorname{softmax}_{j\in\mathcal E_k}(C^M_{j,k}/\tau).
\]

Then

\[
L_{relation}=\frac{1}{K_{eligible}}\sum_k\operatorname{JSD}\left(q_{ref}(\cdot\mid k),q_\theta(\cdot\mid k)\right).
\]

A one-evidence window has a degenerate distribution and contributes zero JSD. Windows without evidence are excluded. This loss preserves the fixed reference's relative relation structure; it does not reward uniformity or maximize entropy.

## 11. Residual-effect distillation

Use the last language-model hidden layer and mean-pool over the tokens of `R_k`:

\[
\Delta h^M_{j,k}=\operatorname{mean}_{t\in R_k}h^{M,full}_t-\operatorname{mean}_{t\in R_k}h^{M,-j}_t.
\]

The auxiliary loss is

\[
L_{residual}=\frac{1}{|\mathcal E|}\sum_{j,k}w_{j,k}\operatorname{Huber}\left(\Delta h^\theta_{j,k},\operatorname{sg}(\Delta h^{ref}_{j,k})\right).
\]

The Stage-1-initialized reference and Stage-2 policy use the same Qwen architecture and hidden width, so no projection layer is introduced. The residual is an effect to match; it is never subtracted from the normal reasoning representation. A residualized-prediction ablation is outside this implementation.

## 12. Objective and scheduling

The Stage-2 objective is

\[
L_{S2}=L_{CE}+\lambda_{EW}(r)L_{EW}+\lambda_{rel}(r)L_{relation}+\lambda_{res}(r)L_{residual},
\]

where `r` is normalized training progress. Each auxiliary multiplier supports a configurable CE-only warmup and linear ramp interval: it is zero before its warmup boundary, rises linearly from zero to the configured coefficient across its ramp interval, and remains at that coefficient afterward. A zero-length ramp switches directly at the boundary. No fixed research-specific ratios are hard-coded. All new coefficients default to zero, which must avoid both auxiliary computation and auxiliary logging dependencies.

## 13. Configuration surface

Names should follow the existing argument/config style, but the functionality must include:

- ILVR stage and required Stage-2 initialization checkpoint;
- enable/disable EW and `lambda_ew`;
- `beta`;
- `loo_intervention = window_attention_block`;
- `contribution_reduction = mean | sum`, default `mean`;
- exhaustive edges or `max_evidence_edges_per_window`;
- `reference_mode = none | precomputed | online_frozen`;
- reference checkpoint and artifact path;
- enable/disable relation KD, `lambda_rel`, and temperature;
- enable/disable residual KD, `lambda_res`, and Huber settings;
- CE-only warmup and per-loss ramp settings;
- contribution threshold and optional full-matrix dump interval.

The old zero-latent intervention may remain isolated as a historical ablation utility, but it is not a supported fallback or alias for the new Stage-2 exact intervention. Source edits must not be required to switch supported experiment variants.

Configuration validation is explicit:

- `reference_mode = none` selects reference-free EW and cannot enable relation or residual KD;
- DPO, relation KD, and residual KD require either a matching precomputed artifact or the supported online frozen mode;
- residual KD additionally requests last-layer hidden states only for auxiliary forwards;
- `online_frozen` with ZeRO-3 is rejected before model duplication;
- an auxiliary enable flag with a zero coefficient performs no auxiliary compute and emits a warning or is normalized to disabled consistently.

## 14. Diagnostics and logging

At minimum log:

- `loss/ce`;
- `loss/ew_pref`;
- `loss/relation`;
- `loss/residual`;
- global diagnostic margin and outer sigmoid/DPO coefficient;
- `ew/mean_contribution`;
- `ew/positive_edge_ratio`;
- `ew/num_edges`;
- aggregated reasoning-window contribution summaries;
- top-1 evidence contribution mass per window;
- normalized contribution entropy;
- number of evidence spans above a configured threshold.

For concentration diagnostics, contributions within each eligible window are converted to the same temperature-controlled softmax distribution used by relation analysis; top-1 mass and normalized entropy are computed from that distribution. Threshold counts operate on raw contribution values. The global diagnostic uses the configured aggregate margin `M` and logs `beta * sigmoid(-beta * M)` as the outer preference coefficient.

Large matrices are not logged every step. Full relation matrices are optionally dumped at a configured interval with sample keys. Concentration and entropy are diagnostics only and must not be described as proof of evidence use or as optimization targets.

## 15. Test-driven implementation milestones

### Milestone 1: explicit stages and parser

Write failing tests first for:

- Stage-1 loss/masks matching the pre-change implementation;
- Stage-2 features-off path using self-generated latent states and CE only;
- absence of helper-image/EMA/similarity work in Stage 2;
- Stage-2 checkpoint requirement and loading;
- exact `R_1,Z_1,R_2,Z_2,R_3` parsing;
- first/last spans, padding, adjacent markers if legal, empty windows, and heterogeneous batches;
- token parser and preserved `sequence_plan` agreement.

### Milestone 2: exact attention intervention

Write failing CPU/synthetic tests showing that blocking `R_k → Z_j`:

- changes only the intended query-key entries;
- leaves `R_<k` outputs unchanged within tolerance;
- preserves access to every `Z_l`, `l != j`;
- preserves positions, length, causal masking, and padding;
- slices correctly in the Stage-2 prefix loop;
- rejects an unsupported backend instead of zeroing latents.

### Milestone 3: contribution and preference

Test window-only log-prob aggregation, mean/sum modes, full-versus-LOO contribution, no-edge reductions, the saturation-gradient property, finite loss, and gradients through both current-policy full and LOO branches.

### Milestone 4: fixed reference

Test artifact identity validation, DPO arithmetic, frozen reference parameters, no reference gradients or optimizer state, ZeRO-stage safety checks, and unchanged reference parameters after an optimizer step.

### Milestone 5: relation KD

Test masked per-window softmax, JSD symmetry/finite behavior, one-evidence windows, variable candidate sets, and gradients only through student contributions.

### Milestone 6: residual KD

Test exact window pooling, full-minus-LOO direction, Huber calculation, artifact dimensions, student gradient flow, and stopped reference targets.

### Milestone 7: integration and smoke

Run the new CPU/unit suite, relevant existing ILVR tests, repository lint/static checks, and one proportionate B200 validation matrix covering:

1. Stage 1 one-batch training;
2. Stage 2 baseline one-batch training;
3. Stage 2 reference-free EW forward/backward;
4. precomputed-reference EW-DPO;
5. relation KD;
6. residual KD;
7. finite losses and intended gradients;
8. frozen-reference no-gradient behavior;
9. features-off baseline reproduction.

The expensive validation matrix is run once after cheaper failures are eliminated. Existing authenticated results are not rerun merely to produce a second manifest.

## 16. Expected code responsibility boundaries

Use existing repository naming conventions while preserving these responsibilities:

1. Stage-specific trainer/collator selection;
2. interleaved span parsing and edge generation;
3. window-local attention intervention construction;
4. matched auxiliary forward execution;
5. token/window score aggregation;
6. EW preference/DPO computation;
7. reference artifact loading and validation;
8. relation and residual losses;
9. schedules and metrics;
10. focused unit and integration tests.

Do not place all logic in one trainer method and do not refactor unrelated ILVR code.

## 17. Complexity and memory

For `E` selected evidence–window edges, the first exact policy implementation requires approximately one auxiliary full forward plus `E` pair-specific LOO forwards, in addition to the ordinary CE forward:

\[
O(1+E)\text{ auxiliary forwards},\qquad E\leq JK.
\]

Counterfactual edges may later be batched along the batch dimension without changing semantics. FlexAttention or layer-local replay may be evaluated only after equivalence with the SDPA oracle is established.

Frozen teacher/reference forwards use `torch.no_grad()` where valid. Precomputed artifacts are preferred for 7B ZeRO-3 training. Existing mixed precision, gradient accumulation, and checkpointing behavior should be reused. A second full model copy under ZeRO-3 is prohibited.

## 18. Backward compatibility and migration

- Stage 1 remains the existing node-grounding training path.
- Stage 2 gains an explicit baseline rather than inheriting Stage-1 helper supervision.
- All new Stage-2 loss coefficients default to zero.
- Features-off Stage 2 does not execute SDPA auxiliary or load reference artifacts.
- Existing zero-latent EW utilities may remain for historical comparison, but they are not exposed as the exact Stage-2 method.
- Existing evidence-window configs must be reviewed and either mapped explicitly to the historical ablation or replaced with clearly named Stage-2 configs; no ambiguous silent reinterpretation is allowed.

## 19. Scientific claims and limitations

Documentation and comments must state that:

- LOO contribution is not a total causal effect;
- it estimates conditional/direct dependency under retained history and other evidence;
- independent edge losses intentionally differ from a global preference objective;
- sparse evidence use can be correct;
- high entropy is not inherently better;
- attention weights alone do not establish utilization;
- single-evidence LOO cannot identify all synergy or redundancy.

Pair/group removal and Shapley-style attribution are out of scope. Interfaces may permit future intervention groups, but the initial implementation supports single-evidence LOO only.

## 20. Completion report requirements

The final report must include:

1. files changed;
2. the actual implemented objective;
3. exact ILVR definitions of `Z_j` and `R_k`;
4. exact attention-block semantics and backend;
5. additional forward and memory complexity;
6. configuration examples for every implemented variant;
7. exact verification commands and summarized results;
8. reference and teacher gradient/memory behavior;
9. known limitations;
10. deviations from this specification and their rationale.

Completion may be claimed only after fresh verification evidence for the final code state is inspected.

## 21. Self-review checklist

- The design resolves the observed shared Stage-1/Stage-2 trainer ambiguity by defining and testing an explicit CE-only Stage-2 baseline.
- The intervention changes only `R_k → Z_j` attention and has no zeroing/deletion fallback.
- Full and LOO contributions use a matched backend.
- Reference meaning is distinct from EMA teacher meaning.
- ZeRO-3 does not duplicate a full 7B reference model.
- Mean window log probability is the default and sum is explicit.
- Exhaustive correctness precedes approximation.
- No-edge and heterogeneous-batch cases are specified.
- Relation and residual targets share the same Stage-2 self-generated latent semantics.
- New objectives default off and Stage 1 remains unchanged.
- Scientific interpretations are bounded and do not equate density or entropy with correctness.
