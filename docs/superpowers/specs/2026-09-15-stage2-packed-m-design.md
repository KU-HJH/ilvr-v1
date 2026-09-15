# Stage 2 packed M suffix replay — design proposal v2

Status: **design commitment v2 approved by the user ("ㅇㅋ"), 2026-09-15**.
The historical proposal/approval-boundary wording below records the pre-approval
sequence; approval applies only to sections 1–6 plus section 8, unchanged.

understood as: leave normal Stage 2 latent generation unchanged; eliminate
blockwise full M rollouts by sharing safe F prefix K/V and grouping independent
M suffix predictions. The user confirmed this scope and authorized a CPU
premise probe. The approval target will be the explicitly numbered final
commitment appended after critique synthesis, not this proposal alone.

Normative objective and acceptance criteria remain
`documents/ILVR_Codex_Superpowers_Implementation_Prompt.md`. This document only
changes the execution of M, not the loss, windows, intervention or training
baseline. Working source baseline: `2dc0e01` in `stage2-entropy-vision`.

## 1. Decision and alternatives

Recommend **safe prefix sharing + packed independent suffix replay**. Reuse
the current runtime/window selection, current Qwen decoder and its DynamicCache
interface; use ours1's branch visibility pattern, not its objective or boundary
zeroing. No runtime dependency on the ours1 worktree.

Rejected alternative: batch K full sequences. It groups calls but keeps
repeated prefix/latent work and multiplies active sequence storage.

Deferred alternative: first implement K independent cached suffix calls.
Useful as a comparison oracle, but not the desired default M execution because
it retains K prediction invocations. The existing full replay remains the
primary equivalence oracle and explicit compatibility mode.

Prediction calls, **excluding unchanged latent recurrence**, when blocks exist:
CE-only=1, entropy-only=2, Vision-only=2, combined=3. M runs once per microbatch,
not once per block. No claim that a whole latent model executes only twice.

## 2. Normal paths stay unchanged

- Do not modify the normal latent rollout algorithm or its call schedule.
- F remains the same full supervised forward: marker CE, latent-pad ignore,
  text-only entropy renormalization, full-vocabulary CE/S_F and latent gradients.
- Reference remains a separate frozen Stage 1 model with its own normal Stage 2
  latent rollout. No reference cache, EMA, teacher injection or alignment work.
- Stage 1, feature-off legacy, enabled CE-only, and entropy-only do not acquire
  any M capture/replay work. Zero valid blocks skip capture and M.
- Do not change tuning targets, optimizer, data, tokenizer, sampling, windows,
  auxiliary normalization, loss formulas, weights or distributed reduction.

## 3. Branch layout and cache ownership

For each existing valid WindowBlock, replay the **contiguous** original token
range `[end_position, max(post_positions)+1)`. This includes END and all
teacher-forced tokens needed before the last selected predictor, even when the
target selector skips a token. Project only selected predictor rows for S_M.

Here `s_k = block.start_position` is the original latent-start marker index;
`e_k = block.end_position` is the latent-end marker index. Slots are exactly
`s_k < j < e_k`. The start marker at `j == s_k` remains accessible. Each
`post_positions` entry is an original full-sequence predictor-logit index, not
a packed query index. Existing selection/validation defines the same targets.

Capture actual F final-forward K/V by layer, detached, with original mRoPE.
Capture is observational: no changed F inputs, stochastic draws, output values
or autograd edges. Reusing normal latent values *before the intervention* does
not change how those values were generated.

Concatenate branch query ranges in one decoder input. Keep original position_ids
for every query, including repeated positions if ever present. Use a fresh
DynamicCache for M, populated from detached captured K/V; never mutate F storage.
Its physical cache/query positions are not replacement mRoPE coordinates.

Let T be the original padded input length. Root cache keys use physical
indices `[0,T)`. Pack blocks in the existing selection order, with
`L_k = max(post_positions)-e_k+1`, `o_k = sum(L_h for h before k)`, and
`Q = sum(L_k)`. Query `q=o_k+u` represents original position `e_k+u`.
Its local suffix key `v` has physical index `T+o_k+v`. Create the 4-D
additive mask `[1,1,Q,T+Q]` before execution and pass
`cache_position=arange(T,T+Q)` to the fresh cache. Pass separately the original
three-axis position_ids gathered at `e_k+u`; do not recompute RoPE from these
physical cache positions. Keep root padding in place but inaccessible. A suffix
key is allowed only when its original attention-mask entry is active.

For branch k every layer/head permits:

- Root keys `j < e_k`, except the current slots `s_k < j < e_k`, and never padding.
- Its own causal suffix keys `T+o_k+v` for `0 <= v <= u` only, subject to the
  original padding mask; no other M branch's keys.
- No root key at/after its current END. In particular, no clean F END/post cache
  access. Previous blocks' normal history remains accessible for this branch.

The shared root can physically contain states that a particular branch cannot
read; the per-query mask must make those inaccessible. The current END is
recomputed from its unchanged token embedding, never zeroed or replaced by F's
contextual END state. Truncating work after the last target is causal pruning,
not deleting tokens or shifting the logical sequence/prefix/positions.

First version packs all valid branches into one M call. Record actual query/key
sizes; do not silently subsample blocks or switch to a different objective.
No automatic OOM retry or generalized scheduler. If representative memory use
requires chunking, report it and obtain approval for the changed call-count
contract; do not hide a K-call fallback under a 2-forward label.
The trigger is an observed memory/resource conflict on an explicitly requested
real workload, not an invented benchmark or a hypothetical OOM. This version
does not define or promise a maximum supported sequence length beyond existing
input limits; it does not claim a new memory-capacity result.

## 4. Dropout and gradient safety

Keep DoRA dropout **0.05**. Do not use eval-mode M or disable dropout.

During the unchanged final F decoder call, record each supported adapter dropout
site's pre-call RNG state, shape, dtype, device and layout. For M, reconstruct the
same full-shape dropout mask under the saved RNG state, then select the original
token rows corresponding to the packed queries. Reuse that mask on M activations,
not F activations. Restore outer CPU/already-initialized CUDA RNG states. Merely
reseeding a shorter packed tensor is not equivalent.

Use narrow, exception-safe capture/replay contexts scoped to this forward only.
The prototype demonstrated temporary hooks and dropout method replacement; the
production version must restore exact module attributes and remove every hook
before returning, including on exceptions. No persistent global ablation flag,
no modified installed PEFT, no changed latent engine. Validate the known single
active unmerged LoRA/DoRA layout and fail explicitly for unsupported stochastic
sites/layouts. Full tuning with zero native attention dropout needs no adapter
dropout replay. No hidden fallback that changes train/eval or dropout probability.

The initial supported adapter layout is the repository's allowed language
projection suffixes with one active, unmerged standard Linear LoRA/DoRA adapter.
Each active training `nn.Dropout` site is keyed by module identity and final-F
invocation index; the supported final-F invocation count is exactly one per
site. A repeat is rejected, not overwritten. Record only final F, not latent
prefix calls. Support rank-3 contiguous `[1,T,features]` dropout inputs, ordinary
non-inplace `nn.Dropout` with `0 <= p < 1`, and record exact dtype/device. Sites
that are identity/eval-mode do not become stochastic in M. Reject multiple active
or merged adapters, nonstandard language stochastic sites, incompatible input
layout/count, nonzero native attention dropout, or mismatched F/M site identities
in packed mode. Preserve existing behavior in full_replay mode. CUDA handling
must use the actual site's device generator and must not initialize unused
devices; it remains unverified until CUDA execution evidence exists.

M, cache extraction and mask reconstruction run without a backward graph. F
backward and native non-reentrant checkpoint recomputation run with original
modules and masks; capture state is not active during recomputation. Preserve
parameter state-dict names and checkpoint parameter identities.

## 5. Integration and configuration

Add `stage2_m_execution = full_replay | packed_suffix`, default `full_replay` to
preserve existing invocation/resume behavior. The three objective examples
explicitly select `packed_suffix`; it only does work when lambda_vis>0 and
valid blocks exist. Expose the choice through existing CLI/coordinator forwarding,
not a new launcher or orchestration system.

Save the execution choice in objective metadata. An older objective config with
the field absent means `full_replay`; do not silently migrate a resumed job to
packed execution. Reject an explicit resume mode mismatch. Keep reference
identity and all existing strict resume checks. No unrelated schema framework.

Within Stage2ObjectiveRuntime, compute the packed per-block S_M scores once and
use them in the existing per-block softplus reduction. Do not materialize a
full-sequence vocabulary tensor per M as the prototype's adapter API does.
Keep current distributed block means/accumulation behavior unchanged. Execute
graph-free M with the existing unwrapped-student path, not DDP forward wrappers.

Counters distinguish actual model calls, decoder calls, M branch count and M
query rows. `model_forwards_M` must not pretend K top-level calls occurred when
the packed path directly invoked one decoder. Empty ranks still participate in
the existing fixed metric schema and reductions.

For packed execution define `model_forwards_M=0` (no top-level model call),
`decoder_forwards_M=1`, `m_branches=K`, `m_query_rows=Q`; all are zero when M is
skipped. Preserve existing actual top-level/decoder F/reference counts and
full_replay M counts. The user-facing prediction count is the final F decoder
call + packed M decoder call + final reference decoder call if enabled, not
the sum of top-level model counters. Observe decoder execution at runtime.

Expected production touch points:

- New `src/stage2/packed_m.py`: narrow capture, visibility, dropout replay/scoring.
- `src/stage2/objective_runtime.py`: routing and per-block scores.
- `src/stage2/objective.py`, `src/utils_deepseed.py`: execution config/validation.
- `src/stage2/trainer.py`: counter schema only, if needed.
- `src/stage2/reference.py`: explicit missing-mode compatibility in objective
  identity comparison, not reference loading/generation changes.
- Existing Stage 2 launcher/coordinator forwarding and objective example script.
- Focused Stage 2/multinode tests and `docs/stage2-objective.md`.

The CPU prototype used the current Transformers fork unchanged. Prefer retaining
that interface; no generated-model regeneration or wholesale ours1 transplant.

## 6. Evidence, acceptance and limits

Authorized CPU premise probe result:
`results/packed-m-probe-20260915/verification.md`; executable throwaway probe at
`.superpowers/probes/2026-09-15-packed-m/probe.py`; full successful output in
`logs/packed-m-probe-20260915/04-compare-verified.log`, exit 0.

16 actual tiny-Qwen comparisons (2 layers, 2 blocks, active DoRA .05):
eager/SDPA × FP32/BF16 × checkpoint off/on × post window 1/2.
M logits error <=1.5e-8; observed F logits, loss, 42 parameter gradients and four
latent-state gradients differences 0. Old decoder counts F=5/M=10/ref=5 become
F=5/M=1/ref=5. Selected-latent perturbation leaves the selected M unchanged;
unmatched dropout and exposed F END cache are detected by negative controls.
The initial negative-control detection threshold failure and its correction
are preserved in the report, not relabeled as a successful first run.

This supports CPU feasibility only. Production acceptance must newly test:

1. Actual runtime full/packed logits, per-block S_M, loss, parameter and latent
   gradients under active adapters; no-intervention packed replay matches F.
2. Selected-latent perturbation and wrong-mask/END-cache/cross-branch controls;
   actual images/mRoPE, left/right padding, skipped targets, windows and zero
   blocks. Larger K and longer targets must not introduce extra M decoder calls.
3. Dropout RNG/module cleanup including exceptions; F backward after M and
   checkpoint on/off comparisons. Invalid stochastic configurations fail clearly.
4. Stage 1/legacy/CE-only/entropy-only regression, skip conditions, model state
   serialization, old-config resume and explicit mode mismatch.
5. Native CPU DDP unequal/zero-block ranks and accumulation against the existing
   full-replay oracle, and existing coordinator dry-run forwarding.

Predeclared CPU production comparison tolerances: FP32 `atol=1e-6, rtol=1e-5`;
BF16 autocast `atol=1e-3, rtol=1e-2`, applied to selected M logits, per-block
scores, total loss and parameter/latent gradients. F direct outputs, outer RNG
state and serialized model parameter values must remain exactly unchanged by
capture. Require identical None/non-None gradient sets, finite values and
independently nonzero post-only latent gradients; tolerances cannot substitute
for those checks. Use FP32 deterministic leakage probes with safe M delta
`<=1e-7`, sensitive F delta `>1e-4`, and the intentionally exposed END control
`>1e-6`. Wrong dropout/visibility controls must exceed the relevant equivalence
tolerance. CUDA comparisons/tolerances require a separately authorized GPU
verification protocol; no CUDA correctness acceptance is claimed by the CPU
implementation gate.

Use TDD for production. Retain the full replay oracle. Target the current
eager/SDPA, FP32/BF16, B=1, native non-reentrant checkpoint and native DDP envelope;
do not expand ZeRO/FlashAttention/FP16/FSDP/compile support.

Unvalidated premises: CUDA dropout-mask replay/layout fidelity, actual image
path equivalence, GPU kernel numerical differences, large-model memory and wall
time, and two-host execution. Record those as untested unless actually run.
No speedup multiplier or task-performance benefit is claimed from call counts.

The final commitment will seek approval to proceed from the
CPU premise evidence, **not** a GPU correctness/performance waiver, permission
to deploy, or permission to train. GPU training/smoke remains a separate request
using the mandatory two-node/four-GPU coordinator and canonical baseline. A GPU
support/readiness claim requires corresponding execution evidence.

## 7. Approval boundary

No production changes have been made. After critique synthesis and approval of
the final commitment: writing-plans, TDD implementation through the prescribed
execution workflow, scoped code review and claim-matched verification. This
proposal does not itself authorize implementation or overwrite the kept branch.

## 8. Design commitment v2 — final approval target

**Status: explicitly approved by the user ("ㅇㅋ"), 2026-09-15.** The exact approval target is
**design commitment v2: sections 1–6 of proposal v2 plus this section 8**.
Sections 1–6 specify the problem/scope, final design, interfaces, tests and
support limits. Sections 7–8 describe approval, not an already granted approval.
The earlier proposal labels and scope confirmations do not authorize production
implementation. This numbered commitment is the artifact presented for approval.

Frozen proposal v2 SHA256 before this appendix:
`dca2610ff8941847a986f6616b42fe18ceafe13d822b630fcf7bf2f969be83ce`.

Changes from proposal v1: explicitly define start/end/slot coordinates, packed
root/query/cache indices and padding visibility; specify supported dropout
invocation/layout guards, actual top-level-vs-decoder counters, CPU tolerances,
the observed-resource-conflict trigger for any later chunking decision, and the
versioned approval target. These resolve the single context-free handoff read.
The scope remains unchanged normal latent generation + packed independent M.
No changes to frozen proposal v2's design were made during critique synthesis.

The repeated hate critique identified stochastic equivalence as the single
load-bearing objection. Its proposed first nail is the active-DoRA full/packed
score/gradient comparison with a deliberately unmatched-dropout control.
Status: **covered on CPU by the already authorized/executed 16-case probe**;
not run on CUDA. Exact observations and initial test failure history are in
`results/packed-m-probe-20260915/verification.md`.

Prism's causal/numerical lens supports proceeding to production implementation
with the listed tests; its execution-cost lens leaves GPU memory/time unclear.
The unresolved question is deliberately retained: fewer decoder calls do not
yet establish faster or lower-memory GPU training. No averaging or consensus
claim resolves it. Macrothink was not invoked because there is no unsettled
direction choice after the user's repeated scope confirmations and the direct
CPU premise probe. Detailed advisory findings and the handoff-read response are
in `results/packed-m-probe-20260915/design-critique.md`.

Remaining unvalidated premises are exactly those in section 6: CUDA stochastic
and numeric fidelity, real image path, larger-input memory/time, and two-host
execution. The CPU probe is not independent task-performance ground truth and
does not claim task improvement. No GPU correctness/performance risk waiver has
been granted or requested. Approval accepts implementing and testing the stated
CPU-backed design while these limits remain explicit; it does not authorize
training, remote deployment, changing dropout, or silently expanding support.

After approval, proceed to writing-plans and TDD implementation. Any later
material change to this commitment's scope, behavior or interfaces returns to
the design gate.
