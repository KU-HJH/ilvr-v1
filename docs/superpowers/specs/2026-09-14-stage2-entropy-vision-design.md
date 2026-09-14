# Stage 2 entropy transition and counterfactual Vision gain

Status: design commitment v1 approved by the user's “ㅇㅋ” on 2026-09-14.
Date: 2026-09-14.

Understood as: implement the fixed three-term objective in the current main
Stage 2 self-generated-latent path, without reusing ours1 V-off, changing Stage 1,
or redesigning the research objective. The user's “그대로하자” confirms this
direction; this document is the final written-design review artifact required by
the requested brainstorming workflow and standing AGENTS commitment gate.

Normative requirements: `documents/ILVR_Codex_Superpowers_Implementation_Prompt.md`.
Sections 1–10 below are frozen as design proposal v1. Review findings and the
implementation-authorizing commitment will be recorded separately below them.

## 1. Runtime baseline and scope

Inspection baseline: main `c21b4e5c311204df4281317219b3f4a6b95e4d3b` plus existing
user worktree changes. Those changes must be preserved. No ours1 preference,
V-off sidecars, prefix caches, teacher helpers, EMA, alignment, RL/DPO, additional
KL, block sampling, or unrelated infrastructure will be imported.

The active path is `src/main.py:build_stage2_training_components` and
`Stage2Trainer.compute_loss` in `src/stage2/trainer.py`, not the Stage 1
compatibility import in `src/trainer.py`. Current Stage 2 already rejects helper
image tensors and injected teacher latents. It uses original images, a frozen
visual encoder, and its existing trainable language/adapter parameter policy.

Observed imports with the repository venv and explicit PYTHONPATH are
Transformers 4.50.0.dev0 from `transformers/src/transformers`, PyTorch 2.7.1+cu128,
PEFT 0.17.1, Accelerate 1.10.1. Runtime Qwen changes live in
`transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`.
Although that file has a generated-file header, its ILVR Stage 2 customizations
are absent from the modular file. Patch the maintained runtime hooks narrowly;
do not regenerate and overwrite the ILVR customizations.

The present rollout in `stage2_latent_rollout.py` recomputes the full prefix for
each latent slot, reinserts its final hidden state without detaching, and makes
one final full-sequence decoder pass. Retain that execution structure. Actual
model/collator microbatch support remains one; pure window/mask helpers support
batched inputs. Reject unsupported runtime batch sizes explicitly.

## 2. Objective and configuration

For each block in the same final valid set K, average entropy within each window
before forming ratios:

```
r_pre  = (H_theta_pre + eps) / (H_ref_pre + eps)
r_post = (H_theta_post_F + eps) / (H_ref_post_F + eps)
D = r_pre - r_post
loss_dyn[k] = -sigmoid(D - gamma_H)
S_F = mean_token log p_F(target)
S_M = mean_token log p_M(target)
G = S_F - stop_gradient(S_M)
loss_vis[k] = softplus(m_V - G)
loss_total = loss_ce_F + lambda_dyn * mean_k(loss_dyn[k])
                      + lambda_vis * mean_k(loss_vis[k])
```

Add Stage 2 CLI/config options with the specification's defaults:
`enable_stage2_objective=false`, `lambda_dyn=0`, `lambda_vis=0`,
`reference_model_path=unset`, `gamma_H=0`, `m_V=0`, `entropy_eps=1e-6`,
`pre_window=1`, `post_window=1`.
Reject nonfinite numeric options, negative weights, nonpositive eps/windows,
noninteger windows, enabling outside Stage 2, missing reference when dyn is
active, and disabling marker supervision in the new path.

Feature off dispatches to legacy without adding reference, masks, or new CE
semantics. Feature on with both weights zero uses the existing self-generated
F forward only; its supervised CE and gradients remain the baseline. Handle
all-ignored labels as a graph-connected finite zero only in the new path.
Do not change normal CE reduction. Dyn zero skips reference construction and
forward; vis zero skips M. Valid K empty skips auxiliary forward work and
contributes graph-connected auxiliary zero, while retaining CE.

## 3. Tokenizer, windows, and labels

Observed expanded Stage 1 tokenizer encodes `<|latent_start|>`, `<|latent_end|>`,
and `<|latent_pad|>` as individual tokens; obtain IDs from the actual loaded
tokenizer/config, not these observations or constants. Assert agreement with
Stage 1 metadata and reference. Keep marker CE and ignore latent-pad CE.

Construct a recorded vocabulary partition from actual tokenizer IDs, special
tokens, and known structural token strings. Exclude padding, chat/modality
controls, image/video/latent placeholders, and structural tool/FIM tokens even
when the tokenizer does not mark them special. Ordinary added text tokens stay
included. Record exclusions and reasons, and output rows without mapped tokens.
Fail on an unresolvable tokenizer/output-row mismatch instead of inventing IDs.
The inspected base tokenizer has more output rows than token IDs; do not assume
`len(tokenizer)`, `tokenizer.vocab_size`, and model output width are identical.

Entropy is `-sum(exp(log_softmax(z[V_T].float())) * log_softmax(z[V_T].float()))`
using natural logs. CE and both scores normalize over the full output width.
Never modify CE logits in place. Select window rows before FP32 conversion;
avoid auxiliary full-sequence FP32 vocabulary copies.

Feature-specific collator metadata identifies the actual assistant text
segments and token bounds from the rendered/tokenized plan. Do not alter prompt
bytes, image order, trajectory, or label rules. Verify offsets against final
tokenization; reject ambiguous alignment rather than silently assigning text
to a different segment. No truncation is introduced.

For start position s and end position e, pre window consists of the last
`pre_window` text-state logit positions in P, including s-1; it does NOT test
whether the next label is text. Post window is the first `post_window` valid
supervised text targets i in the immediately following assistant text segment,
using logits i-1. These exact targets define both S_F and S_M. Preserve skipped
tokens in the teacher-forced sequence. Do not cross turns, latent blocks,
padding, or truncation boundaries to fill a window.

Collect all candidate windows, then remove any logit position appearing in a
pre and a post window from both sides, simultaneously; do not refill or iterate
until overlaps disappear. Remove blocks whose resulting pre or post is empty.
Use these identical final masks for student, reference, and both auxiliaries.
Strict malformed-marker errors retain sample/block context. Well-formed blocks
without adequate text are excluded with explicit reason counts. Padding never
becomes a target or window state. Reject nonfinite intermediate values with
sample/block/location diagnostics; do not clip D or replace NaNs with zeros.

## 4. Independent M branch and gradients

F runs once in student training mode, with gradients through original F CE,
entropy, S_F, and every self-generated latent. Each M_k independently reruns
the current student from the original processed input, under no_grad, without
using any F latent embeddings, hidden states, or suffix caches. Previous latent
blocks remain normal; subsequent latent generation is recomputed under k's
intervention, not under additional interventions. Reference generates its own
latents independently.

M preserves input IDs, original image tensors, marker embeddings, slot counts,
sequence length, teacher-forced text, and original position IDs/RoPE. In every
decoder layer/head, add the following restriction to the original causal and
padding mask:

```
block attention(q, j) iff q >= e_k and s_k < j < e_k
```

It applies at latent_end itself and in each applicable prefix/final pass.
Position calculation continues to use the original 2-D padding mask. Pass an
explicit immutable intervention descriptor through the Stage 2 rollout and
materialize a correctly sliced additive decoder mask per invocation. Never use
a mutable ablation flag, slot deletion, zero embeddings, or a normal suffix
cache. The tensor mask passed into checkpointed decoder calls is captured as
that call's input, so later M calls cannot change F backward recomputation.
Padded query handling must not unmask a restricted edge for any valid query.

M stays in the same training mode/backend/autocast policy as F: no_grad is not
eval. Capture CPU/CUDA RNG immediately before F; replay it for each M in a
scoped RNG context and restore the state after F. This requires identical call
shapes/order and is tested with the existing DoRA dropout 0.05. Do not silently
disable DoRA dropout or globally switch student to eval. If replay does not
control a combination, reject it; supporting it requires a reviewed design
change. Reference eval and loading must not perturb the student's RNG stream.
Detach S_M again at the objective boundary as an explicit safety contract.

## 5. Frozen Stage 1 reference and resume

Load a separate model from the specified original completed Stage 1 checkpoint
using the existing full-weight or base-plus-DoRA loader, including schema-2
sparse token rows. Do not deepcopy a training student, load a resumed student
as reference, share parameter storage, or add reference parameters to any
optimizer. Freeze every parameter, set eval, and execute under no_grad.
Use identical processed inputs and compatible vocabulary/backend/dtype; compare
student/reference token rows at matching Stage 1 initialization. Training-mode
student dropout means D need not be zero at initialization; require D near zero
only under matched deterministic evaluation conditions.

Save resolved original reference source, artifact identity/checksums,
tokenizer/vocabulary-mask identity, and complete objective configuration with
the existing Stage 2 checkpoint metadata. Resume reconstructs reference from
that source and verifies identity; reject missing/changed artifacts or an
incompatible objective override. Preserve saved sampler/resume settings.
Hash once at loading/verification rather than per microstep; tests compare actual
reference parameters across student optimizer steps and reloads. No entropy
cache or general artifact/provenance system is introduced.

## 6. Reduction and metrics

Auxiliary mean is over valid blocks across all ranks and all microbatches of
one optimizer accumulation window, not a mean of local/rank means. Determine
the final valid-block metadata for the upcoming window through the actual
Trainer `get_batch_samples` hook and all-reduce its block count N on every rank,
including zero-block ranks. For DDP world W and Trainer loss division A, each
microbatch's differentiable auxiliary sum is scaled by W*A/N before backward;
DDP's /W and Trainer's /A then recover the global block mean. N=0 uses zero.
Account explicitly for the Trainer's actual last partial-window scaling. Keep
the existing CE reduction, including its remainder behavior, unchanged.

This formula is conditional on runtime scaling: tests must witness native
Trainer/Accelerate and any admitted DeepSpeed path; reject incompatible custom
accumulation plugins instead of assuming another hidden divisor is absent.
No parameter-forward collective is added by M on the supported DDP path.

Log count-weighted, globally reduced unscaled means separately from the scaled
backward scalar: CE, dyn, vis, total; student/reference pre/post entropy; both
ratios and D; S_F/S_M/G; r_pre>1, r_post<1, G>0 fractions; valid/excluded blocks
and reasons; reference token-window occurrences whose entropy is below eps;
actual extra model/decoder forward counts and backend. Low-reference-entropy
counts are occurrence counts across selected pre/post windows, with units
documented; they do not modify the loss. Disabled metrics are marked absent,
not fabricated zeros that suggest reference/M was evaluated.

## 7. Support and cost boundary

Initial auxiliary correctness targets: eager and SDPA, microbatch one, no KV
cache or effective sliding-window attention, zero native attention dropout,
FP32 and BF16 autocast, existing full/LoRA/DoRA parameter policies, native
non-reentrant gradient checkpointing. Reject FlashAttention2 for the auxiliary
path and ZeRO-3 before model loading. Retain feature-off and zero-weight legacy
backend choices where they need no intervention/reference. No automatic backend
fallback. Zero-weight CE still obeys existing backend/ZeRO restrictions.

DDP is the initial distributed target. ZeRO-1/2 are conditional targets: enable
only with actual multi-forward/scaling validation; otherwise reject them for
active auxiliaries and report not supported. Do not describe CPU mocks as GPU,
ZeRO, or real-data support. Native attention dropout and custom checkpoint/
accumulation combinations outside these conditions fail explicitly.

For L total latent slots and K valid blocks, F costs L prefix decoder calls
plus one final call. Active reference adds one full rollout; active Vision gain
adds K full rollouts, each no_grad. M can be scored and released sequentially.
An online reference adds separate model storage per rank; do not assume a
replicated model fits. Prefix sharing, truncation, caching, and scheduling are
deferred optimizations requiring equality evidence and separate scope approval.

## 8. Minimal change surface

- `src/utils_deepseed.py:parse_args`, `src/main.py`: opt-in
  options, validation, lazy reference construction, and Stage 2 wiring only.
- `src/stage2/data.py`: feature-only assistant segment/window metadata.
- Focused new helpers under `src/stage2/`: objective math/window vocabulary,
  fixed reference loading/identity, and accumulation metadata as needed.
- `src/stage2/trainer.py`: F/reference/M orchestration, RNG scoping, loss
  integration, count reductions, metrics, existing save/resume metadata hooks.
- Qwen runtime modeling and `stage2_latent_rollout.py`: explicit intervention
  propagation and selected-row access; preserve legacy signatures/defaults.
- `tests/stage2/`: focused unit tests and actual tiny-Qwen/Trainer integration.
- One opt-in execution example script and documentation using the existing
  multi-node coordinator, without rewriting user-modified launch infrastructure.

Implementation starts in an isolated worktree after final approval. Snapshot
only necessary authorized dependencies from the dirty checkout deliberately;
never reset, stage, or commit unrelated user changes. Use repository venv tools.

## 9. Acceptance and evidence plan

TDD: write each causal failure test first, observe its failure, then implement
the minimum fix. Acceptance groups A–E in the normative spec remain binding.

- Independent hand-computed scalar/logit oracles: renormalized text entropy,
  full-vocabulary scores, D=0 => dyn=-0.5, vis=log(2), derivative signs,
  stop-gradient, per-block nonlinearity before mean, extreme/BF16 values.
- Exact eight-token alignment, marker CE, padding, multi-block overlap removal,
  assistant segments, malformed pairs, empty validity, all-ignored CE, batch
  helper support and runtime batch rejection.
- A deterministic multi-layer causal fixture with deliberately informative
  generated latent values: F sensitivity is the positive control; M invariance
  when those values change is the negative control. Test end-mediated leakage,
  later latent blocks, input mutation, positions, and independent interventions.
  Perturb the generated latent, not just a pad embedding that rollout overwrites.
- Actual tiny-Qwen full/adapter forwards: F post loss reaches latent generation;
  M/reference create no backward graph; checkpoint on/off gradients match even
  after M executes; controlled-dropout no-intervention replay equals F.
- Reference has no shared parameter storage/optimizer ownership and stays
  unchanged after optimizer step and resume; mode-specific forward-call spies.
- Native Trainer integration with irregular valid counts, empty rank/window,
  accumulation and final remainder; compare gradients/updates against an
  independently assembled global scalar oracle. Preserve Stage 1, feature-off,
  and same-forward new CE-only regression witnesses.

Already observed, before this implementation: a tiny main SDPA eight-token
forward matched manual shifted CE and gave nonzero gradients to both generated
latents from a post-only loss. Four existing Stage 2 routing/frozen-visual/
checkpoint tests passed. These establish only baseline behavior; none proves
the new mask, loss, resume, distributed reduction, or GPU support.

No new objective tests or training have run. GPU/real-data checks require the
actual completed Stage 1 source available on both nodes. Local original
checkpoint-431 currently exposes rank-2/3 resume shards rather than complete
rank-0 adapter/tokenizer artifacts; remote availability has not been verified.
Do not substitute an older checkpoint or evaluation-staging copy silently.

## 10. Delivery and experiment boundaries

Deliver patch, options, focused test evidence, design and execution plan, and
three reproducible commands using the SAME Stage 1 source/data/seed:
CE-only (0,0), CE+entropy (positive dyn,0), and CE+entropy+Vision (positive,positive).
Accept paths and auxiliary weights as arguments/environment variables; validate
their existence and avoid pretending example weights are tuned. Follow current
training defaults: four ranks/two nodes, microbatch 1, accumulation 2, global
batch 8, LR 5e-5, seed 42, warmup 10. Derive Stage 2 updates from retained data
and actual Trainer behavior, not the Stage 1 value 431. Preserve phase budgets.

Use existing multi-node coordinator after required operator-guide preflight;
no single-node training fallback. Model artifacts stay under checkpoints,
metrics under results, logs under logs. No full training, remote push, or large
download without separate authorization. Report CPU contracts, tiny actual
model/Trainer checks, four-rank GPU smoke, ZeRO checks, and task evaluations
separately, including commands and not-run reasons.

D increasing does not guarantee both desired entropy directions; stop-gradient
does not preserve M performance. G can increase because S_M drops. This tests
access to the current latent path, not semantic visual correctness, novelty,
stability, or task improvement. Do not equate auxiliary decrease with task gain.

## 11. Commitment-gate review of frozen proposal v1

Self-review corrected the parser owner to the observed
`src/utils_deepseed.py:parse_args` before the critique passes. No production
files were edited. The following advisory passes do not establish runtime
correctness and do not replace implementation tests or human approval.

### hate

```
{
  root: "The design depends on M differing from F only by the specified access
         restriction. Uncontrolled DoRA randomness or reuse of downstream F
         latent states would invalidate G even if the attention-mask unit test
         passes.",
  first_nail: "On a tiny two-block, multi-layer Qwen with baseline DoRA dropout,
         replay F under no_grad with no intervention and compare selected
         logits/RNG state. Then perturb the actual generated selected latents:
         F must respond and independently replayed M must remain invariant,
         including latent_end and downstream latent generation. Run M before
         checkpointed F backward and compare F gradients with the F-only run."
}
```

First-nail status: proposed, NOT RUN. It requires adding intervention/test
instrumentation and is deferred to the first authorized TDD implementation
step, before broader integration or GPU training. No new implementation-like
premise experiment was authorized or executed during this design phase. The
uncertainty is accepted only for reversible implementation work if the human
approves the commitment below; it is not waived for declaring support or
launching training. A failure returns to the active Superpowers design or
debugging gate; it does not authorize a substitute ablation.

### macrothink

Applied because implementing the wrong counterfactual would invalidate the
entire objective integration. Two fresh same-model read-only agents received
the stripped fixed goal, constraints, and runtime facts, without proposal v1
or a preferred solution. They did not edit files or invoke paperthin hooks.

- Divergent-incompatible: none reported.
- Divergent-compatible, backend scope: both preferred beginning with eager
  only, rather than assuming SDPA mask equivalence. Keep proposal v1's explicit
  backend test gate: eager is the first correctness witness; SDPA is admitted
  only after its own actual-Qwen causal/checkpoint/dropout tests pass.
- Convergent: both independently chose separate M trajectories, downstream
  latent regeneration, explicit intervention state, stochastic control, and
  optimizer-window global block sums. This is reassurance, not verification.

Their static observations do not demonstrate executed call counts, dropout
equivalence, gradient scaling, or mask behavior.

### prism

Two distinct lenses read the same frozen proposal:

- Causal validity: unclear — matched-RNG no_grad replay and blocked-path
  invariance in the actual adapter/checkpoint path remain untested.
- Execution feasibility: unclear — replicated online reference plus per-block
  full replay has no measured four-rank memory/throughput witness yet.

Shared verdict: uncertainty remains for different load-bearing reasons. There
is no manufactured disagreement to resolve by averaging. Neither critique
requires changing the fixed loss or intervention. Implementation acceptance
depends on the first-nail witness; training feasibility separately depends on
the required two-node preflight and actual memory/step smoke. Critique agreement
is not premise evidence, and neither gate substitutes for the other.

## 12. Design commitment v1 — approved for implementation

This is the exact implementation authorization proposal. Sections 1–10 of
this document define its detailed interfaces and acceptance conditions; the
[normative requirement file](../../../documents/ILVR_Codex_Superpowers_Implementation_Prompt.md)
remains binding. Proposal v1 is unchanged in problem, scope, behavior,
interfaces, or assumptions by the review. Review clarified test order only:
eager first, SDPA after its own witnesses. No loss changes were proposed.

Implement the fixed CE + entropy-transition + Vision-gain objective as an
opt-in extension of current main Stage 2, without ours1 V-off code. Preserve
Stage 1, feature-off behavior, original-image inputs, marker CE, parameter
policy, and self-generated-latent CE-only behavior. Reference is a separately
loaded, frozen, eval/no_grad model from the original Stage 1 checkpoint, with
its own latent rollout and resume-verified identity. It is absent when dyn is
zero; M is absent when vis is zero.

F denotes the normal student branch; M is its selected-block counterfactual.
s/e denote latent_start/latent_end positions and j a key position. Dyn/vis are
the entropy-transition/Vision-gain terms; D/G are their contrasts defined in
section 2. Section 3 defines the selected slots' neighboring text windows and
simultaneous overlap removal precisely.

F keeps its full gradient graph. Each M independently replays original inputs
with no gradient, recomputing later latents, and blocks every decoder-layer
edge from queries q>=e to selected slots s<j<e. Preserve markers, images,
sequence, RoPE, and targets; use no F suffix cache or latent-state reuse, no
deletion/zero replacement, and no mutable ablation mode. Replay paired RNG
while retaining student training mode and baseline DoRA dropout. The explicit
mask must survive checkpoint recomputation without mode contamination.

Use text-vocabulary-renormalized FP32 entropy at s-1 and e by default, preserve
full-vocabulary CE/scores, and compute each block's sigmoid/softplus before
the global valid-block mean. Use identical final windows after simultaneous
overlap removal. Aggregate auxiliary block sums over ranks and the complete
optimizer accumulation window with framework scaling verified against an
independent oracle; keep legacy CE weighting unchanged.

Support is evidence-gated, not granted by this document. Target eager, then
SDPA, microbatch one, FP32/BF16, existing full/LoRA/DoRA policies, native
non-reentrant checkpointing, and DDP. Active auxiliaries reject FlashAttention2,
ZeRO-3, effective sliding-window attention, and unsupported stochastic or
checkpoint combinations. ZeRO-1/2 remain disabled unless actual scaling and
multi-forward execution are tested. Preserve existing backend restrictions for
feature-off/zero-weight execution without imposing unnecessary auxiliary limits.

Material divergence: fresh reviewers preferred eager-only initially; resolved
by beginning with eager and admitting SDPA only after separate evidence, as
already required in proposal v1. No incompatible direction or unresolved
reviewer disagreement was reported. Unvalidated premises remain: paired RNG
equivalence under the real adapter/checkpoint path, causal M invariance, native
distributed/accumulation scaling, reference resume invariance, and four-rank
memory/real-data feasibility.

First nail is the CPU tiny actual-Qwen paired replay/latent-perturbation/checkpoint
test specified in section 11 of this document, using randomly initialized tiny
weights rather than a downloaded checkpoint. It does not launch training or
require a two-node GPU job. It has not run because its instrumentation is
implementation work gated by this approval. Approval explicitly permits that
reversible TDD work while retaining these uncertainties; it does not waive any
acceptance test or authorize an alternative objective, unverified support, or
long training. If the first nail disproves the implementation assumption,
resolve it before expanding integration; a material design change needs new
approval. GPU work still requires original checkpoint availability, two-node
preflight, and the repository's existing coordinator.

After approval: writing-plans, isolated implementation worktree, failing tests
first, implementation with the installed subagent-driven workflow, focused code
review, and claim-specific completion verification. Deliver minimum patch,
settings, tests, docs, and three runnable same-Stage-1-source/data/seed multi-node
examples: CE-only, CE+entropy, and CE+entropy+Vision. Providing these examples
does not authorize or imply execution of their full training jobs.
Report not-run GPU/ZeRO/training checks separately. Do not claim D/G improvement
as task-performance improvement. Existing user edits remain untouched.

### Written-commitment comprehension check

The root invoked shower for the durable approval artifact. One context-free
reader received the complete section-12 commitment text, not the repository
or session reasoning. Verdict: minor gaps, with the intended approval and
evidence boundaries understood. Findings were missing navigable references,
undefined shorthand in the excerpt, ambiguity about CPU first-nail versus GPU
training, and runnable examples versus executed jobs. The root clarified those
points above without changing proposal v1's behavior or acceptance criteria.
This check covered the commitment excerpt, not an independent end-to-end
review of sections 1–11 or a runtime validation.
