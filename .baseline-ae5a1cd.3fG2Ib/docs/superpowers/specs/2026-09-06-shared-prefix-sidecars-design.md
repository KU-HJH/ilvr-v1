# Stage 2B shared-prefix sidecars

Status: design commitment v2 approved by the user's “ㅇㅋ” after clarification
that three logical groups do not mean exactly three decoder calls. Tasks1–4,
whole-change review, final scoped fix review and verification are complete.
Long real-input8192 capacity and full training quality remain unverified/out of scope.
Measured evidence: `results/shared-prefix-implementation-20260905/summary.md`.

Understood as: revise the existing `ours1` Stage 2B computation to share clean
decoder prefixes across all local entropy and rejected-scoring branches, while
preserving its current objective, reference conditioning, and training cadence.

Source: `/root/workspace/ILVR/documents/TODO-ours1-20260906.md`.
Implementation worktree: `/root/workspace/ILVR/.worktrees/ours1`.
Audited code baseline: `0958a7688fe9f38f56b79a74deccc0b203eb9713`.

Terminology: V-off blocks this target's prediction queries from the immediately
preceding visual latent pad K/V; T-off blocks those queries from the immediately
preceding text K/V. Rejected origin names the modality disabled during offline
generation, not during DPO scoring. Qwen mRoPE uses temporal/height/width axes.
A latent block is complete only after its latent-end control token.

## 1. Scope and existing behavior

Stage 1, ordinary Stage 2, offline negative generation, and the existing
CE/DPO/entropy formulas are retained. No training experiment resumes as part of
design preparation. Implementation includes small correctness and performance
checks; a full CE/DPO/entropy training campaign is a separate experiment.

The runtime starts at `src/main.py`, joins negatives in `src/stage2/data.py`,
and reduces Stage 2B scores in `src/stage2/trainer.py`.
`LatentRolloutEngine.rollout` performs one no-cache decoder prefix call for each
latent slot. A final decoder call then processes the completed embeddings.

Naive Stage 2B repeats this rollout and final decoder for each entropy/rejected
branch. Entropy forwards carry one step's shifted attention mask; rejected
forwards carry clean context. The existing packed path runs one clean rollout
and a final tree containing the clean root and all sidecars. Its root is already
shared in the final decoder: the proposed change must not claim that this
existing tree repeats a full root per branch.

The new path exposes clean final-pass layer K/V for grouped replay. It removes
the naive branches' repeated clean-prefix projections and rollouts. It does not
change the clean rollout algorithm, whose internal prefix repetition remains
measured separately. Improvement over the existing packed implementation is an
empirical question, not a promised consequence of three execution groups.
Here, "one rollout" means one invocation of `rollout`, which still contains
multiple internal decoder calls; it does not mean one Transformer pass.

## 2. Approaches and selected design

1. **Explicit differentiable K/V, clean + V + T (selected).** Capture the clean
   final decoder's K/V, then process V and T suffix groups against those tensors.
   This fits the requested stages and permits deterministic branch chunking.
2. **Layer-interleaved clean/V/T processing.** Also shares projections, but
   interleaves execution groups and makes the existing decoder loop and
   checkpoint lifecycle harder to reuse. Not selected for the initial change.
3. **Further optimize the current monolithic tree.** Retained as a comparison
   baseline; it does not provide the requested grouped replay implementation.

One outer policy `model(...)` call may own all three logical stages so PEFT,
Trainer, and DeepSpeed retain their usual outer forward boundary. The design
does not impose three Python `model.forward()` calls. Calls through the decoder,
layers, and wrapped projection modules must remain observable and preserve
their framework hooks.

## 3. Clean state and grouped branch execution

The clean stage performs the existing self-generated latent rollout once, then
one clean final decode. It returns clean CE, chosen scores and entropies, and a
microbatch-local state containing completed input embeddings, logical mRoPE
positions, valid root positions, and each layer's post-RoPE K and unexpanded V.
K/V are tensors in the policy autograd graph, not detached inference caches.
Their parameterization and GQA head geometry match the existing attention.

Use explicit immutable tuples of tensors as layer inputs/outputs. Do not update
a shared `DynamicCache`, store cache in mutable global/module state, or populate
cache through a forward hook. Capture K/V as outputs of the layer computation
so activation checkpoint recomputation has the same functional inputs/outputs.

Each sidecar group flattens only its branch query/input rows. At a layer:

- Project Q/K/V and run the MLP only for those rows.
- Attend to the one clean K/V bank and the group's branch K/V with a rectangular
  visibility mask. A temporary concatenation of the bank and group K/V is
  allowed; copying the full prefix into a batch entry for every branch is not.
- Keep the clean bank read-only. Branch state never updates clean recurrence or
  another branch. Branch-local positions identify ordering within each suffix.

The V group contains V-off chosen entropy branches and valid visual-origin
rejected branches. The T group contains T-off chosen entropy branches and
valid textual-origin rejected branches. Ablation ranges belong to individual
entropy branches. Rejected branches have no blocked modality range.

Dense explicit masks are the correctness backend. After dense equivalence,
extend the existing FlexAttention integration to rectangular Q/KV and validate
the same branch visibility on CUDA. Ordinary clean language/vision backend
selection remains as configured; production currently uses language SDPA and
vision FlashAttention2. No dependency upgrade or external trainer replacement.
Rectangular Flex implementation and measurement are required delivery items.
Dense-only success is an intermediate correctness milestone.

## 4. Boundary queries, positions, and local intervention

For chosen target tokens `[a,b)`, scoring queries are `[a-1,b-1)`. The entropy
branch replays input rows `[a-1,b-1)` from clean completed embeddings. Its clean
ancestor bank includes positions strictly below `a-1`. The boundary row at
`a-1` is recomputed under its own mask; its clean hidden state/logits are not
reused. A latent-end control token at this boundary is replayed as an ordinary
boundary input, not used to generate another latent.

Only prediction queries for this step are ablated. V-off excludes root latent
pad K/V; T-off excludes root previous-text K/V. Previously mediated information
in clean embeddings or earlier states remains, as required for local marginal
intervention. The ordinary scaffold has a completed latent block before the
target. Replays must not enter an unfinished latent block; malformed metadata
retains a clear validation error.

For rejected completion token positions `p`, start at
`r = min(clean/rejected token longest-common-prefix length, p[0]-1)` and replay
through `p[-1]-1`. The clean ancestor bank ends strictly before `r`; any shared
boundary/control rows after `r` are replayed. This preserves tokenizer boundary
differences and pairs whose first target token differs. Do not share current
chosen target hidden states even when chosen/rejected tokens initially match.

Carry logical positions independently of packed storage. Chosen replay gathers
the original three-axis positions. Rejected replay uses positions from its own
fully encoded clean-prefix/completion input, computed with the existing
`get_rope_index`. Shared ancestors must have matching tokens and positions.
Rejected text longer than the clean root must not index nonexistent root
positions. Padding never contributes scores or visibility. Current data errors
for empty/unscoreable targets remain errors; zero valid negative pairs remain
legal and contribute no DPO loss. A sample without reasoning steps follows the
existing clean-only reduction where that sample is otherwise valid.

The independent local oracle must not use the new packed planner or predicate.
It uses full prefix recomputation, a directly constructed causal mask with one
shifted rectangle, and per-step truncated inputs. Audit it against the current
naive path before relying on it. Fix a demonstrated local-semantics bug first,
with its causal test and recorded reason.

## 5. Objective, gradients, checkpointing, and resource limits

Reuse the trainer's reduction: per-pair independent DPO sigmoid, valid-pair
mean across modalities, and token-weighted entropy margin across both
modalities. Preserve the existing CE/DPO/entropy coefficients (including current
ablation experiment configurations), clean entropy gradient, and latent/prefix
gradient. Only the clean stage contributes CE.

Preserve production DoRA dropout 0.05. The current packed path already shares
one stochastic clean rollout/root between branches. Preserve that sharing
topology and the adapter modules' ordinary dropout behavior. Independent naive
forwards resample their prefixes; they are a strict numerical oracle only with
dropout disabled. Neither matching seeds nor matching marginal score
distributions proves equal stochastic DPO/entropy objectives. Record this
limitation explicitly. Training-mode tests additionally witness active adapter
dropout, finite gradients through shared state, and RNG preservation during
checkpoint recomputation; do not silently zero dropout or add an RNG framework.

Reduce full-vocabulary branch logits to mean log-probability or per-target
entropy inside a checkpointed replay/scoring function. Returning reduced values
alone does not free saved autograd intermediates. Non-reentrant checkpointing
must cover branch decoder, LM head, and entropy/log-probability computation;
otherwise all groups can still retain full branch activations until backward.
Reuse the repository checkpoint policy; avoid unnecessary nested layer replay
when the entire branch computation is already checkpointed. Prefix tensors are
explicit checkpoint inputs and must receive accumulated branch gradients.

Sum the entire microbatch objective and use Trainer's one backward. Existing
gradient accumulation and optimizer-step timing remain unchanged. No group
backward, optimizer update, `detach`, or prefix `no_grad` is introduced.

Deterministic chunking processes branches in step/modality order under a
configurable query-token budget. It retains every valid branch and the original
global loss denominators. A single oversized branch is processed alone and
reported; token-wise branch splitting is not part of the first implementation.
Chunking/checkpointing is not asserted to guarantee that every 8192-token input
fits. Measure the complete forward/backward peak, including clean rollout,
prefix banks, padding, temporary concatenations, and LM-head computation.

Expose the new path with an explicit opt-in execution option. Keep existing
naive and legacy packed commands available for regression and comparison.
Legacy packing flags and their 4608 packed-admission protocol retain their
meaning. The new shared path uses the existing clean input length limit and its
own branch budget; it must not silently discard samples using the old combined
tree cap. Never reuse legacy packed capacity evidence as proof of new-path
capacity. Existing ordinary Stage 2 and invalid-option validation remain tested.
"Global" denominators here mean all applicable pairs/tokens within the current
microbatch, exactly as `_reduce_stage2b_objective` currently computes them;
gradient accumulation and distributed reduction are not redefined.

## 6. Reference conditioning and artifact use

Observed code: `score_step_completion` invokes the frozen Stage 2A model with
token/pixel inputs in inference mode, and that model generates its own latent
trajectory. `generate_negative_rows` stores those scalar scores. Stage 2B reads
the stored scalars; it does not inject changing policy latents into reference.
Preserve this independent-reference-latent objective. The older modality spec's
same-policy-latent proposal is not the current implementation and is not adopted
as an optimization change.

Cache eligibility requires unchanged reference checkpoint, tokenizer/template,
rendering, tokens, image inputs, positions, latent generation, and numerical
settings. Existing manifests authenticate source JSONL, checkpoint files,
tokenizer/config, rendering, and settings; do not claim they also authenticate
every external image byte. Audit actual consumed inputs and recompute a small
frozen-reference score witness before accepting cache claims. If an input is
not fixed, keep reference scoring online in no-grad under the same reference
conditioning; do not change to policy-generated latent conditioning.
Determine fixed inputs from the actual scoring call's token, pixel, position,
configuration and checkpoint inputs, not simply the checkpoint's frozen flag.
An invalid/missing cache blocks consumption. Any needed online scoring is an
explicit runtime choice with reported calls/cost, not a silent recovery path.

Count reference work separately: actual online model/decoder calls and valid
cache entries consumed/missing. Current static artifacts can yield zero online
reference calls only after the fixed-input audit. Missing or invalid artifacts
must not be counted as hits or replaced with zero scores.

## 7. Verification and measurement

Use the repository venv and vendored Transformers. Reuse existing tests and
measurement scripts; add only the new computation's causal coverage.

Required witnesses:

- No-op replay matches clean logits under equal conditioning.
- Two or more decoder layers and multiple steps: branch changes cannot affect
  clean state, other steps' sidecars, or future/other-branch visibility.
- First target log-probability and entropy respond to the correct local mask;
  include a latent-end boundary and a first-token-different rejected pair.
- An independently constructed dense oracle verifies allowed/blocked edges;
  rejected branches see clean context. Check every layer where practical.
- FP32/dropout-off logits, scores, entropies, each objective component, total
  loss, and every trainable parameter gradient satisfy `atol=1e-5, rtol=1e-4`.
  Report maximum errors and near-zero gradients explicitly. BF16/backend
  differences are reported separately, not judged by an assumed tolerance.
- Prefix/latent gradients, immutable K/V across V/T and recomputation,
  checkpoint on/off, ragged padding, invalid negatives, empty targets,
  feature-off behavior, and the current DoRA/ZeRO-2 two-rank accumulation path.
- Runtime projection and MLP hooks demonstrate suffix-only processing in
  sidecar groups. Separate checkpoint recomputation from initial forward.

Benchmark identical inputs, loss weights, precision, valid tokens, and tuning
method across naive, shared 3-group, and existing monolithic packed where each
fits. Separate compile/warmup from measurement. Report throughput, complete
forward+backward time, peak allocated memory, decoder and policy calls, latent
rollout calls, reference calls/hits/misses, checkpoint recomputation, branch
chunks, projection tokens, padding, and LM-head rows. Change K once at fixed
total target length and once with growing length to avoid conflating effects.

For cost accounting, entropy has `2 * sum(chosen target length)` scored tokens;
valid rejected tokens and boundary/non-target replay are separate. The boundary
prediction row replaces the last target input row for ordinary teacher forcing;
do not mechanically charge one extra row when it is already counted.

Existing historical benchmark numbers do not establish performance on this
new path or new code revision. Unavailable or OOM cases are reported as such,
with no extrapolated completion or performance claim. Consider an optional
clean + all-sidecars 2-group merge only after 3-group correctness and measurement.
An unavailable required check, including two-rank DoRA/ZeRO-2, leaves that
delivery item incomplete and prevents claiming full task completion or enabling
the new route for that unvalidated production configuration. This task does not
include merge or worktree cleanup.

## 8. Expected file boundaries and delivery

- `src/step_preference/packing.py` and a focused `sidecar.py` module: branch
  metadata/group planning and rectangular visibility, retaining legacy APIs.
- Vendored Qwen `modeling_qwen2_5_vl.py` and a focused shared-state helper:
  explicit differentiable K/V outputs/inputs and grouped replay/scoring.
- `src/stage2/trainer.py`: new route and existing loss reduction.
- `src/step_preference/config.py`, `src/utils_deepseed.py`, `src/main.py`,
  `run_stage2_training.sh`, and the collator where necessary: opt-in routing
  and branch budget without reinterpreting legacy admission.
- Existing equivalence/benchmark modules and scripts: independent oracle,
  measured counters, same-input comparisons, and concise result output.
- Focused Stage 2/step-preference tests and current distributed smoke tests.
- The three source specs live only in the primary checkout's `documents/` and
  are currently untracked user files. Apply narrowly scoped documented edits
  there for forward/cache claims; do not silently copy or stage all user files.

Those three files are `ILVR_stepwise_codex_implementation_spec_ko.md`,
`ILVR_stepwise_modality_dpo_spec_ko.md`, and
`ILVR_stepwise_method_problem_spec_ko.md`. Existing execution entry points to
extend are `scripts/verify_packed_step_preference.py` and
`scripts/benchmark_packed_step_preference.py`. Exact new CLI names/defaults and
test cases will be recorded in the implementation plan after approval; the
existing clean limit is 8192, distinct from the legacy packed limit of 4608.

After design approval, write the implementation plan connecting these exact
interfaces and tests, implement via TDD, review, and run claim-specific
verification. The final report includes changed files, reproduction commands,
test outcomes, numerical errors, actual costs, and remaining limitations.

## 9. Design evidence and commitment review

Baseline audit run: `shared-prefix-audit-20260905t160215z`.
Existing tests, without source changes:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_rollout.py tests/stage2/test_stage2_modeling.py \
  tests/step_preference/test_ablation.py
```

Result: 36 passed in 2.14s; complete output is in
`logs/shared-prefix-audit-20260905t160215z/baseline-tests.log`.
An existing two-step fixture with two decoder layers, FP32 and dropout disabled
produced 40 decoder calls in naive mode (32 latent rollout, 8 final) and 5 in
legacy packed mode (4 latent rollout, 1 final). First-layer Q projection input
rows totaled 312 and 152 respectively. Both scalar total losses were
5.562499523162842. The packed length was padded to 128, so these tiny runs are
call-graph observations, not a throughput benchmark. They do not compare the
unimplemented shared-prefix path. Full output is in the adjacent
`call-profile.log`.

The permitted external references were inspected as local source:
`/root/workspace/ILVR/Reference/dpo-prefix-sharing/modeling/dpo_flex_attn_masks.py`
for visibility predicates and `Reference/trl/trl/trainer/dpo_trainer.py` for DPO
scoring/reference handling. Their trainers and dependency versions are not
imported into ILVR.

Critique synthesis before approval:

- `hate` root: reusable final-decoder K/V must preserve the independent local
  branch's multi-layer prediction and gradient graph, including its first
  prediction row and clean latent ancestors. A scalar-loss-only or single-layer
  test could miss invalid state sharing.
- Proposed first nail: a two-layer, two-step FP32 dense comparison with directly
  written local masks, a latent-end boundary, first-token-different rejected
  completion, and every trainable gradient, with checkpointing off/on. Add
  projection witnesses demonstrating that branches do not reproject ancestors.
  Status: not run on the proposed implementation, which does not yet exist.
  Run as the first implementation gate after approval. It is distinct from the
  completed baseline audit; no costly model experiment has been authorized by
  a design-review result.
- `macrothink`: two independent fresh reads identified the same compatible
  gap: grouping does not bound saved autograd memory, and nonzero dropout needs
  an explicit claim boundary. Whole-branch scoring checkpointing addresses the
  first concern. Proposal v2 adds the observed 0.05 dropout and limits strict
  naive equality to the requested dropout-off witness. A third read was
  unavailable because the selected model was at capacity. Agreement is advisory,
  not correctness evidence.
- Both fresh readers then read frozen proposal v2 and reported the stochastic
  scope concern resolved, with no incompatible design finding. These are two
  read-only opinions; the new implementation still requires runtime evidence.
- `prism`: correctness is unclear until the local replay/gradient first nail;
  resource feasibility is unclear until complete forward/backward measurement.
  Both lenses reject claiming completion from three groups or reduced outputs
  alone. Neither finding requires a new objective or sampling policy.
- `mandela`: a shared planner used to generate both candidate and expected
  masks would be circular evidence. Use independent dense masks and direct
  runtime hooks. These checks establish computational behavior, not scientific
  improvements in reasoning quality or negative correctness.
- `shower` cold-read the approval summary and identified terminology, replay
  cutoff, rollout/call distinction, Flex scope, denominators and unavailable-test
  reporting as comprehension gaps. The detailed spec already uses clean K/V
  strictly below each rejected branch's own replay start `r`; that remains the
  governing rule. The other scope/terminology details are now explicit above.
  This was a summary comprehension review, not code or runtime verification.

Unvalidated premises: new-path gradient/checkpoint equivalence, memory and
speed on real long samples, rectangular Flex behavior, and DoRA/ZeRO-2 runtime
integration. No performance, full-dataset OOM, stochastic-naive equivalence, or
reference same-policy-latent claim is waived into existence by design approval.
Approval accepts reversible implementation with these explicit test gates;
failed gates prevent production rollout and require a targeted fix or revised
design. The source TODO's human approval gate was satisfied by the user's
“ㅇㅋ” following the three-logical-groups clarification.
