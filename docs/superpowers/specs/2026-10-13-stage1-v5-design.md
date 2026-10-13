# Stage1-v5: DIL-BCE-ORIG, V-off entropy and Text-off matching

Date: 2026-10-13. **Design commitment v1; final human approval pending.**
Workspace: `/root/.codex/worktrees/6b9d/ILVR`, branch `Stage1-v5`.
Product base: `9119b55cc717d8dfe9fc047cad128d90074e3faa`.
Prior evidence: `../2026-10-13-stage1-v5-workflow.md`.

## 1. Intent, approved direction and scope

Train Stage1 freshly from original Qwen2.5-VL-3B-Instruct with its helper-latent
teacher forcing and alignment retained. Add the exact token DIL objective from
`documents/dil-bce-orig-v1.md`, independent V-off conditional entropy, and
Stage1-v3-style Text-off consistency. Omit Gain, its extra clean graph, explicit
Gain VJPs, projection and optimizer modification. Complete the implementation,
then one fresh two-node training run and automatic final-model TEST400 evaluation.
No accuracy improvement or speedup is promised by this design.

The user selected Subagent-driven development, explicitly retained alignment,
set `0.1*(CE+alignment)`, selected packed OFF branches for four decoder calls,
set k=50/entropy weight=0.1, requested Stage1-v3 as the Text-off source, and moved
work to Stage1-v5. The latest `진행해` approved writing the revised design after conversational
section review. This commitment will bind only after final written human
approval; that approval permits planning, followed by written-plan review.
Training and automatic evaluation authorization already exist.

This supersedes old Stage2 integration Tasks2-6. Their numerical Task1 kernels
and independent review are retained unchanged. Standard Stage1, Stage1-v3 in its
own worktree, and existing Stage2 selectors/resume semantics stay separate.
The original DIL source file remains byte-identical.

## 2. Source reconciliation and chosen approach

The DIL source assumes Stage2B. Stage1-v5 uses its mathematics and auxiliary
target contract within Stage1, without student latent autoregressive rollout.
Its reference is the initial effective Stage1 teacher, not a trained Stage1
checkpoint. This explicitly replaces the source's arbitrary separate reference
checkpoint loader with the equivalent immutable initial-teacher artifact for
this Stage1 profile. Moving EMA is rejected.

Stage1-v3 source is pinned to commit
`e555a2b6ddf43d689cceca7342fa3fb6a17b90c3` in
`/root/.codex/worktrees/541e/ILVR`. Its inspected file hashes are in the workflow
record. Port the needed metadata, selection, tracing, replay and KL behavior
into this worktree; never import the other checkout at runtime. Its Gain
controller/lifecycle cannot be copied unchanged: it owns different graphs and
backward phases, and its RV branch also masks text.

Two approaches were presented: separate OFF calls (five decoder calls), or a
packed pair (four calls). The user chose the packed pair. Reusing helper-pass
logits as DIL reference is excluded because that pass precedes latent injection
and does not evaluate the same clean history.

## 3. Four-call data flow

At microbatch 1 with eligible auxiliary targets:

| Call | Model and inputs | Outputs and use |
| --- | --- | --- |
| 1 Helper | Existing frozen Stage1 teacher, existing helper-target procedure | Select helper visual features once; detach and inject the same latent snapshot into every subsequent context |
| 2 Clean F | Live student, original tokens/images and selected helper latents | Existing CE/alignment; selected clean logits for DIL and detached Text-KL target; attached prefix K/V and detached attention observations captured during this call |
| 3 Reference | The frozen teacher again, now with exactly F's helper-latent snapshot and clean token/image/position/attention history | Detached full-vocabulary probabilities at the same predictors; used for DIL and Text-off KL selection |
| 4 Packed OFF | Live student decoder, separate rows per block and per intervention, each with its own slice of F's clean prefix | V-off logits for entropy and optional mass; Text-off logits for matching KL |

This is five semantic contexts in four decoder invocations, not four equivalent
units of FLOPs. Vision encoder calls and checkpoint layer recomputation are
additional. Do not advertise four calls for batches with no auxiliary targets;
those retain ordinary helper/F work and graph-connected zero auxiliary losses.
Actual call hooks, not configuration labels, establish the runtime count.

No separate Gain-clean forward is allowed. Capture F's real prepared embeddings,
position IDs and attached K/V during F. Do not reconstruct its prefix from a
second student pass, detach its prefix gradients, or borrow student K/V for the
reference. All live losses join one ordinary backward and native Stage1 AdamW.

## 4. Data, positions and independent interventions

Preserve existing Stage1 source preparation, chat rendering, tokens, images,
helper selection, CE labels and alignment's existing 0.01 target perturbation.
Use Stage1-v3's verified source-item-to-token mapping as the starting point:
anchor to original assistant content, check token offset roundtrip, and exclude
tokens crossing adjacent source text-item boundaries only from auxiliary spans.
No inserted separators or retokenized training sequence is permitted.

Each source latent block contributes one DIL/entropy step, including an empty
step. Its eligible targets are supervised ordinary tokens wholly inside the
single immediate next text item. Structural tokens, latent slots and padding
are excluded from auxiliary scoring. Predictor is target minus one, including
END as predictor of the first following text token. Preserve empty original
steps in DIL/entropy denominators; Text-KL uses eligible query positions only.

Every intervention starts independently from the clean prefix for that block:

- V-off blocks all current latent-body keys for the replay suffix. Prior text,
  prior latent blocks, current START/END, and original image remain available.
  It does not inherit Text-off's mask or accumulate previous interventions.
- Text-off keeps the current latent body available and masks selected previous
  assistant-reasoning text keys at each scored query. Candidates precede the
  current START; question/prompt, original image, special tokens, current target
  text and boundary-crossing tokens are not candidates.

These are direct attention-access interventions, not deletion/reindexing of
tokens, zeroing latent values, or claims that all indirect information has been
removed from contextual hidden states. Packed rows cannot attend to each other.

## 5. Exact losses and selection

Let logp=log_softmax(F.float()), logmu=stopgrad(log_softmax(reference.float()))
over the entire vocabulary, and f=logp-logmu. At each selected target y:

```text
D_positive = softplus(-f[y])
D_reference = sum_v exp(logmu[v]) * softplus(f[v])
D = D_positive + D_reference
```

Reference includes GT, without 1/V, 1/2, top-k truncation, probability flooring,
or a second independent reference weight. Beta=1 and margin=0 are required.
Use the reviewed `dil_bce_orig_loss.py` implementation and its FP32/checkpointed
position chunks; it explicitly rejects nonfinite normalized log probabilities.

For V-off entropy, detach top-k candidate IDs selected from V-off logits
(`independent`). Normalize only selected logits to q, with both q and logq live:

```text
U = log(number_of_candidates) + sum(q * logq)
```

This minimizes KL(q || uniform) and raises conditional entropy. The run uses
k=50, no GT union, U coefficient 0.1. Do not add the old rejected BCE/softplus.
The numerical support helper also supports explicit `sync_topk` and GT union
for the source contract, preserving actual ragged support sizes.

Optional mass retains the reviewed Bernoulli KL(detached clean support mass ||
OFF support mass), over the exact entropy support. Its full-vocabulary OFF
normalizer leaves complement gradients live. Mass is zero in this run; a
positive configured mass coefficient requires active entropy. Chosen mass is 0.

Text selection follows Stage1-v3's KL mode: detached T=2 full-vocabulary
d = T^2*KL(softmax(F/T) || softmax(reference/T)), transformed by
-log(max(d,0)+1e-6), centered
within sample and mapped to rho=0.3+0.2*sigmoid(centered value). Rank previous
text keys by head/layer-mean attention from the full causal softmax; do not
renormalize over candidates. Select the shortest attention-ranked prefix whose
cumulative mass reaches rho times total candidate attention mass, capped at
floor(0.3*N). If candidate attention mass
is below 0.1, use min(cap,floor(0.1*N)); break ties by original position. Empty
masks are valid. Reuse the DIL reference call, without an extra selector pass.

Text matching is `K = T^2*KL(stopgrad(softmax(F/T)) || softmax(Text_off/T))`,
T=2, using ordinary F as target rather than frozen reference. Preserve the
Stage1-v3 local reduction: query mean within each valid sample, then valid
sample mean within the microbatch; a microbatch without queries contributes 0.
K weight is 0.01 from the first update (no K warmup). LR warmup remains 10.

The requested run is exactly:

```text
L = 0.1*CE + 0.1*alignment + 1.0*D + 0.1*U + 0.01*K
```

The reusable orig selector permits source-compatible entropy off, candidate
mode/GT union, optional mass and detached token-selection settings. Orig token
selection retains the earlier approved definitions: clean or OFF full-vocabulary
entropy, absolute entropy gap, or intersection of the two stable top-percent
position sets without refill. Apply the same position mask to D/U/mass. This
run uses all eligible positions (no percentile selection). Text masking and its
K targets are separate from this orig position selection.

For optional configurations, execute V-off only when entropy/mass or an
OFF-dependent orig position selector needs it; execute Text-off only for a
positive K coefficient. Pack both when both are required. The frozen clean
reference remains required by DIL even when both OFF terms are inactive.

## 6. Reduction and distributed execution

D/U/mass reduce token losses within each original step, then average original
steps globally in the actual accumulation window; empty original steps count
as zero steps. Never average rank means or chunk means as a substitute. Preserve
the existing Stage1 CE/alignment and Text-KL accumulation convention separately.

Let N be the all-rank original-step count for the actual window and W=4. After
all Trainer/DeepSpeed accumulation scaling and rank averaging, the auxiliary
D/U/mass gradient must equal the gradient of sum(all step losses)/N. Derive the
per-microbatch multiplier from the actual parent Trainer and DeepSpeed path;
test the result rather than assuming a declared GAS field is the divisor.
Zero N produces student-connected zeros with matching collective order.

Prefetch/count metadata through the actual Trainer accumulation hook so every
rank knows the same window denominator before backward. This change is opt-in
to v5. A rank with no auxiliary targets may omit local noncollective work but
must still join the same collective sequence; do not invoke wrapped distributed
forwards asymmetrically without establishing the wrapper's actual behavior.

Preserve Stage1's real sampler and its repeats/padding. Do not import Stage2's
zero-loss padding rule or hardcode 212, 216 or 431 updates. Derive one-epoch
budget, actual prepared-loader microsteps, final remainder and observed source
occurrences before launching. CE/alignment/K preserve the configured global-16
denominator on a partial accumulation, with the final behavior verified against
the actual framework. Record how this differs from actual-global-step averaging
for D/U/mass.

## 7. Frozen teacher/reference and saved identity

Fresh student initialization is original Qwen2.5-VL-3B-Instruct, Stage1's three
new tokens and DoRA. The existing Stage1 teacher deep-copies the final effective
student topology before training. Require ema_tau=1, all reference parameters
frozen, every module in eval mode, and all calls under no_grad. Reference outputs
and injected helper latents are detached. No mutable student/reference parameter
storage alias is permitted. Preserve reference-local transient forward buffers
without treating a mutable cache as part of the model's immutable weight identity.

Reuse the same frozen owner for helper selection and clean reference, with a
second decoder evaluation for the latter. Validate tokenizer ID mapping, special
tokens, vocabulary, latent layout, image preprocessing, predictor alignment,
attention and full position IDs. Reference handles its own token/image modules
and its own K/V; only the helper latent inputs are shared by value.

Record an immutable initial effective reference artifact under checkpoints:
original base content identity plus the initial adapter/new-token parameter
state and processor/config identity. Reuse existing Stage1 EMA serialization
where it covers every mutable effective parameter, proving that coverage. Its
path alone is insufficient. No full base copy per student checkpoint is needed.
Every resume must restore this same teacher/reference before any target or logit
forward, verify content identity, and reject a missing or changed artifact.
Never initialize a resumed reference from resumed student weights.

Add a v5 objective/identity sidecar alongside the existing Stage1 PEFT format;
preserve existing model loading and inference preparation. Save formula IDs,
coefficients, masks/selection/reduction semantics, reference identity, prepared
sampler schedule and package/code profile. Omitted resume options inherit saved
values; explicit conflicts fail before training. Standard/v3/orig variants
cannot be silently relabeled on resume. No historic artifact is rewritten.

## 8. Integration boundaries and memory

Use an opt-in `stage1_variant=v5` (launcher spelling `--stage1-variant v5`). Keep
Stage1-specific configuration outside the Stage2-only phase dataclass. Expose
the explicit loss aliases `dil-bce-orig` and `dil_bce_orig`, canonicalized only
through an explicit alias map. Resolve CLI aliases once, reject conflicting
spellings, and retain the source's public top-k/mass names. Do not reinterpret
legacy `dil_bce_rf*` fields or defaults.

The current run's explicit entropy0.1/k50/Text-KL0.01 profile must not be
confused with the standalone orig selector's optional entropy default of 0.
The launch recipe passes the selected values explicitly after sourced settings.

Suggested focused boundaries: v5 config/identity; source-aligned metadata;
Text-off selection and loss; clean trace and packed replay; trainer integration
and reduction; existing coordinator argument forwarding and execution recipe.
Choose exact public option names and produced/consumed types in the approved
implementation plan. No generic recovery, retry, ownership or archival system.

The source's same-history and full-vocabulary requirements are mandatory. Reject
moving EMA, incompatible source/tokenizer, unsupported image replacement/full
block removal, nonzero DIL margin, legacy rejected loss or legacy entropy-margin
mixing. For this production profile support microbatch 1, BF16 DoRA and ZeRO-1;
do not claim other training modes without corresponding evidence.

Reuse position chunking (128) for exact FP32 loss intermediates and selected
reference/OFF projections. Avoid retaining reference logits for every sequence
position when only eligible predictors are consumed. Preserve ordinary clean
CE behavior. Any selected-logit model entrypoint must remain opt-in and leave
legacy paths unchanged. Packed batch rows scale with source blocks; four calls
do not establish memory feasibility for full 3B. Bind the production memory
witness to the actual retained dataset maximum packed geometry: branch count,
padded prefix and suffix lengths, selected query count and vocabulary
projection chunks. The maximum prefix and suffix may belong to different
blocks. Account for the dense replay bias and replicated prefix K/V, not only
the input sequence length. Record the observed geometry and measured peak
memory. A typical record alone does not satisfy this launch gate.

Non-reentrant checkpoint closures must preserve the correct parameter/buffer
bindings and original sample/token dropout identities across F and packed rows.
All F and OFF gradient paths use optimizer-owned student parameters with one
ordinary backward. Preserve gradients through reused prefixes and input token
rows. Port only the relevant v3 execution mechanisms, not its Gain alias/VJP
state machine. No extra optimizer displacement is allowed.

## 9. Required evidence and failure behavior

| Claim | Required witness |
| --- | --- |
| Exact DIL/U/mass mathematics | Reuse reviewed kernel evidence; focused fresh tests include analytic equal-policy gradient and live entropy/complement gradients |
| Stage1 data preserved | Real tokenizer/source mapping comparison, unchanged clean tokens/labels/helper target identity, crossing/empty/adjacent item fixtures |
| Reference same history and immutable | Tiny patched Qwen/DoRA tests with equal weights in eval; differing token embeddings as negative control; shared latent values; frozen effective state after student update and resume |
| Packed equals independent | Literal independently masked full-sequence Qwen oracle, unequal blocks/lengths, first predictor at END, dropout 0 and 0.05, checkpoint off/early/full recomputation, output and parameter/input gradient comparison |
| Single F / four calls / no Gain | Runtime hooks at helper, F, reference and packed decoder entries; one ordinary backward, no Gain-clean or projected update; empty-target behavior and mode restoration |
| Correct actual Trainer reduction | Parent Trainer/DeepSpeed execution witness with unequal step counts, empty rank/sample, GAS4 and partial tail; compare parameter gradients/update to an independently assembled global oracle |
| Resume and CLI correctness | Saved-objective conflicts/missing reference fail before forward; both worker command lines carry identical coefficients and source/reference/schedule identity |
| Production execution | Existing coordinator live preflight and a bounded two-node/four-rank integration witness; finite gradients, immutable reference, observed ranks, counts and measured peak memory |
| Requested result | Final checkpoint plus original base prepared under checkpoints/evaluation-staging, complete TEST400 coverage and evaluation summary |

Fail explicitly on nonfinite values, incompatible history, unsupported masks,
reference mismatch or collective inconsistency. Preserve failure logs and use
one targeted fix/test then a fresh run ID where required. Do not fall back to
single-node training or report partial/skipped tests as full runtime success.

The architectural premise concerns replacing v3's separate alias clean graph
with ordinary F while simultaneously supporting CE/alignment, DIL and attached
packed OFF gradients. The separately authorized tiny CPU probe at commit
`8b675e2` directly exercises that new graph; its result and review status are
recorded below. Prior Stage2 reference replay and v3 tests are not substituted
for this witness. Full Trainer, BF16, distributed and capacity evidence remain
launch gates even if the tiny graph comparison passes.

## 10. Training, evaluation and completion

Use existing multi-node coordinators and repository venv. Reuse reviewed native
worktree root support from Stage1-v3 only where this checkout needs it; do not
invent another launcher or overwrite canonical/other worktrees. Provision a
compatible remote copy and explicitly verify imports point to the reviewed
Stage1-v5 code and patched Transformers on both nodes.

After sourcing settings, enforce: remote rank0 at korea@10.1.102.41/container
llm_kd, local rank1, bond1.102, two GPUs per node, microbatch1/GAS4/global16,
LR5e-5, seed42, warmup10, BF16, AdamW/linear, weight decay0.01, clip1,
DoRA r32/alpha64/dropout0.05 on validated decoder modules, latent8,
frozen teacher1.0, decoder SDPA, vision FlashAttention2, non-reentrant
checkpointing, Stage1 maximum sequence32768, workers0 and ZeRO-1. Record the
resolved profile and actual data-derived budget rather than copying examples.
No run launches before compatible-code/resource/input preflight passes on both
nodes. Four-rank participation needs runtime evidence.

The existing final-artifact preparation and evaluation coordinator run after
successful training/checkpoint validation. A simple detached sequential recipe
is sufficient; a failed training/preparation step must not evaluate an earlier
checkpoint silently. Evaluate TEST400 with greedy temperature0/top_p1,
max_new_tokens4096, max_sequence_length128000, repetition_penalty1,
no vision ablation and external_full_latent_block. Model-bearing outputs stay
under checkpoints; metrics under results; logs under logs, using explicit
experiment/run IDs. Report actual metrics and any missing evaluation coverage.

Completion still requires revised written-spec approval, plan review (the
Subagent-driven choice persists), per-task TDD and independent specification/
quality review, applicable final review, fresh verification and the branch
finishing gate. A merged branch/PR is not implied. No new product integration, GPU training or
evaluation is claimed complete by this design. CPU premise evidence is recorded separately and cannot
establish full production training equivalence or memory feasibility.

## 11. Design commitment v1 and evidence

Frozen input: `2026-10-13-stage1-v5-design-proposal-v1.md`, SHA-256
`71d78c7280d7ca14f5bd28b2c77863958d447b28779936de45b2bbe52d7b893e`,
commit `f4f839fbc11b5dfdfad1496057f7f99951f3ee12`. Root hate and three fresh
macrothink reads assessed these exact bytes; the proposal remains unchanged.

The agreed problem and scope remain Sections 1-2; the final design is Sections
3-10. Changes from the proposal clarify Stage1-v3's T-squared selection KL and
cumulative attention threshold, and bind the later memory witness to actual
retained-data maximum packed geometry. They preserve the requested loss,
four-call structure and independent interventions. The Text-KL diagnostic fix
improved evidence sensitivity without changing that design.

No incompatible divergence was established. Three divergent-compatible reads
contributed those selector/memory refinements and required all five loss terms
to be exercised together through optimizer-owned parameters. Convergence is
reassurance only. Source evidence and classifications are in
`2026-10-13-stage1-v5-critique-v1.md`. No material critique disagreement remains
unresolved within the accepted design/probe scope.

**First-nail status:** the separately authorized CPU probe supports the attached
ordinary-F/packed-OFF graph on its specified tiny fixture; it did not falsify
that premise within the tested scope. This is bounded evidence, not a general
runtime-equivalence proof. Experimental commits are `fe9f434` and `8b675e2`.

- Six comparisons cover dropout 0/0.05 crossed with checkpoint off/early/full,
  actual patched Qwen/DoRA/native RMSNorm, all five terms and one ordinary total
  backward. All 44 trainable tensors, live prefix inputs and a native in-memory
  AdamW update are compared against independently masked full-sequence calls.
- At unchanged atol1e-5/rtol1e-4, maximum selected-logit error is 1.1921e-7,
  loss-term error2.5216e-7, combined-gradient error2.9803e-8 and parameter-update
  error0. Tiny scalar KL agreement is only an absolute-tolerance claim.
- A first detached-prefix control was below tolerance. A deterministic x4
  initial output-head scale made it detectable without changing loss,
  normalization, masks or tolerances. Original failure logs remain preserved.
- Independent review found that omitting the tiny weighted Text-KL escaped the
  original fused checks. The fix compares isolated raw-K gradients in all six
  cases (maximum error7.75e-9); intentional omission fails all six assertions.
  Detaching Text-off produces mismatch2.61e-4 at the same atol1e-5. Diagnostic
  and ordinary-backward recomputation counts are recorded separately.
- Three causal controls reject the intended equivalence assertion: detached
  clean prefix, V-off contaminated by Text-off masking, and detached Text-off.
  They do not establish sensitivity to every possible implementation error.
- Original decoder entries are F/reference/packed. Reference effective state
  stays frozen/immutable through the student update; helper inputs are detached.

The independent task review identified the Text-KL evidence gap; its scoped
re-review closed it. The final experimental review found no remaining
Critical/Important/Minor findings. The durable final review is
`../reviews/2026-10-13-stage1-v5-premise-final-review.md`; exact review history,
agent identities, hashes and commands are in the workflow record.

Fresh root verification after final review: **9 passed in 10.02s**, exit0:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q logs/stage1-v5-premise/test_probe.py
```

Complete output: `logs/stage1-v5-premise/root-final.log`; machine-readable
case evidence: `results/stage1-v5-premise/result.json`.

The helper-fixture probe observes only the three-call F/reference/packed
subgraph. It uses fixed realized input embeddings; each reference has its own
frozen decoder/head weights and K/V, but reference-local embedding-adapter and
vision preparation do not execute. This does not permit borrowing student token
embeddings in production: Section 7 needs its own runtime witness. CPU FP32,
tiny DoRA r4, one sample, fixed masks/support/noise and the coordinate-dropout
harness do not establish production BF16/r32, selectors/data mapping, helper
selection, the full four-call Trainer, distributed reduction, EMA/ZeRO/resume,
capacity or throughput. Those explicit integration and launch gates remain.

This exact commitment v1 is ready for final written human review. Approval
permits writing the implementation plan; the plan must then be reviewed before
Subagent-driven implementation. Existing authorization for the subsequent fresh
two-node training and automatic TEST400 evaluation persists.
