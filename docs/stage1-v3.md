# Stage1-v3

`stage1-v3` adds the text-only restriction objective from
`documents/text_only_gain_v1.md` to the existing Stage 1 path. Select it with
`--stage1-variant v3` on the coordinator/launcher, or `--stage1_variant v3`
on the training CLI. An omitted variant on a fresh run retains `standard`.

The implementation has focused CPU contract tests and an independent review;
final suite evidence is recorded in `logs/stage1-v3-implementation/workflow.md`.
**Distributed acceptance has not been run.** Tiny CPU checks do not establish
four-rank execution, full 3B memory use, throughput, or TEST400 accuracy.

## Profile and objective

Fresh v3 uses the original Qwen2.5-VL-3B-Instruct base, DoRA r32/alpha64/dropout
0.05 with q/k/v/o/gate/up/down decoder projections, latent size 8, CE weight
0.1, alignment weight 0.1, frozen EMA 1.0 and helper noise 0.01. Execution uses
two nodes/four ranks, microbatch 1, accumulation 4, global batch 16, LR 5e-5,
seed 42, warmup 10, BF16, AdamW/linear, weight decay 0.01, clipping 1.0,
ZeRO-1 without offload, decoder SDPA and vision FlashAttention2,
non-reentrant checkpointing, max sequence 32768 and zero data workers.
The full profile is explicitly resolved after sourced environment settings.

Each valid immediate text block has independent R and RV replay branches.
R excludes selected prior text keys at scored queries; RV additionally blocks
the current latent body for the suffix. START/END and the original image stay
available. Source-item crossing tokens are excluded only from auxiliary spans.
The collator's input tensors and CE labels retain their existing meaning.

Text selection defaults to FP32 full-vocabulary student-to-reference KL at
temperature 2, centered within each sample, rho 0.3–0.5, text cap 0.3,
fallback 0.1 and image cap 0. To select by student entropy instead, pass
`--stage1-v3-selector entropy` to the coordinator/launcher, or
`--stage1_v3_selector entropy` to the training CLI; `kl` selects the default.
The option is valid only for Stage1-v3. Entropy uses existing gain-clean query
logits: `p = softmax(logits / 2)`, `H = -sum(p * log(p))`, then
`u = -log(max(H, 0) + 1e-6)` and the same sample-centered sigmoid/rho mapping.
Lower entropy within a sample means more prior text is blocked; singleton or
equal-entropy queries use the rho midpoint. Attention ranking, deterministic
ties, caps and low-mass fallback are shared by both modes. Entropy measures
student uncertainty rather than reference disagreement; no accuracy
equivalence or improvement has been established.

In both modes the ordinary loss adds temperature-2 KL(detached F || R), averaged
by query within each valid sample then across samples. Fresh runs use weight
0.01 from the first step (`text_kl_warmup_ratio=0`). The separate LR warmup
remains 10 updates. Resume restores the saved KL coefficient and warmup ratio.

The fresh Gain default is `adamw_projected`, coefficient 10, with raw token
entropy weighting as described below. In the legacy `post_adam` mode,
R and RV GT scores are block means. Their gradient sums and block count B are
reduced across all four ranks. With g=(R−RV)/B and h=RV/B, a negative global
h·g component is projected away using FP64 dot/norm reductions. Tiny nonzero
h norm squared below 1e-30 produces no displacement. After a successful AdamW
step, alpha=0.001 times the direction is added to FP32 master parameters and
synchronized to BF16; Adam moments remain unchanged by this addition.
An engine-skipped step clears the auxiliary window without applying Gain.

`--stage1-v3-gain-weighting token_entropy` on the
coordinator/launcher, or `--stage1_v3_gain_weighting token_entropy` on the
training CLI, is enabled by default for fresh v3 runs. `none` disables weighting.
For each valid target token, use the ordinary full-context student F logits
at its preceding query position to compute temperature-1 Shannon entropy
`w = detach(-sum(p * log(p)))`, where `p = softmax(F)`. This raw entropy uses
natural logarithms, has no `log(V)` normalization and is not capped at 1.
`--stage1-v3-gain-entropy-mode normalized` selects the historical
`H/log(V)` formula with its [0,1] bound; `raw` is the fresh default. Python
CLI spellings use underscores. Detached FP32 query chunks bound
entropy temporaries; entropy is not computed over the whole sequence.
Multiply each token's `log p_R(GT) - log p_RV(GT)` by this coefficient before
the original token mean within each block and actual-global-B mean. Do not
divide by the sum of coefficients or rescale alpha. Higher full-context
uncertainty gives greater relative contribution; this heuristic has no
established accuracy or vision-use improvement.

The protection gradient h remains the original **unweighted** RV score
gradient. The default optimizer mode below protects its incremental AdamW
displacement; legacy `post_adam` retains its original projection. The joint VJP
carries the already-differenced weighted Gain, so RV is not subtracted again.
Existing F and packed R/RV logits and the same two explicit VJP calls are
reused; no forward is added. This option is independent of the KL/entropy
text selector, which still uses temperature 2, and does not change ordinary
CE, alignment or Text-KL. Call counts do not imply equal checkpoint
recomputation, memory or throughput. CPU tests do not establish this weighted
mode's real GPU or distributed training behavior.

With valid auxiliary blocks, KL selection executes F once, gain-clean once,
frozen reference once and one packed R/RV decoder call. Entropy selection
omits only that reference call: three core calls, or four including the
one helper-teacher decoder forward at the default microbatch 1 (KL: five).
Checkpoint layer recomputation and separate helper-image vision encoder calls
are additional. The packed call contains two branches per valid block, so
the call count is not a measured compute or speed ratio.
Joint two-lane R/KL (weighted Gain/KL when enabled) VJP and a separate RV VJP preserve attached
prefix gradients and original sample/token dropout identity. No valid blocks
means ordinary F/helper execution with zero auxiliary work.

## Optimizer-integrated Gain

Fresh runs default to `--stage1-v3-gain-update-mode adamw_projected` and
`--stage1-v3-gain-optimizer-weight 10`. An explicit coefficient can be supplied
on the coordinator/launcher. Python
CLI spellings use underscores. Fresh environment aliases are
`STAGE1_V3_GAIN_UPDATE_MODE` and `STAGE1_V3_GAIN_OPTIMIZER_WEIGHT`; explicit CLI
values take precedence. This gradient coefficient must be finite and
nonnegative, including zero. It is not inferred
from the legacy displacement `gain_alpha=0.001`. In this mode alpha is
recorded as inactive. Explicit `post_adam` selects the legacy update without
an optimizer weight; supplying an optimizer weight in that mode is an error.

For ordinary gradient b, global-block-mean Gain g and **raw unweighted** RV
h, the native optimizer receives `clip(b - weight*g)`. No raw-gradient
projection is applied to g first. A temporary calculation on owned FP32
shards computes the base AdamW candidate from `clip(b)`, using the same
pre-step parameters, moments, counters, group settings and LR. Each branch
uses its own clipping norm. There is one real native update; its joint
moments/counters are retained exactly.

Only the incremental displacement `joint - base` is projected against h,
using global FP64 dot/norm statistics. Projection excludes groups whose
captured LR is zero and ignores partition padding. A negative RV dot after
FP32 rounding suppresses the whole increment for that boundary. Zero weight
or zero global blocks use the native path without a base calculation or
projection. Successful LR=0 steps still advance native moments; skipped
steps advance neither parameters, moments nor EMA.

The guarantee concerns this step's additional **FP32-master** displacement.
Rejected Gain remains in Adam moments and can affect future steps. Finite
BF16 rounding can reverse directional alignment, and no nonlinear score or
learning-quality improvement is guaranteed. Guards check native masters,
moments and BF16 casts before each group copy, and corrected candidates
before final synchronization. A detected failure forbids checkpointing;
there is no rollback promise after native mutation.

The supported runtime is torch 2.7.1 native fused AdamW, DeepSpeed 0.18.7
ZeRO-1, BF16/FP32 masters, no offload and loss scale 1. Model forward/VJP
execution is reused. CPU tests cover AdamW state/candidate comparisons and
controlled partition/collective traces; actual fused GPU/ZeRO behavior,
additional memory and OOM behavior require separately authorized checks.
Neither GPU checks nor training are part of this implementation task.

Boundary logs under `stage1_v3/` include gradient/displacement norms, RV dots,
per-group captured LR, coefficient and projection/rounding-fallback events.
Counts are per-boundary 0/1 indicators; their observed means give rates.
Unavailable measurements are omitted on bypass/skip and the reason is
recorded explicitly. These diagnostics are separate from ordinary loss.

For an argument-only preview, append the following to the **dry-run command
below**, keeping its `--dry-run` flag. The coefficient 0.25 is illustrative,
not a validated training recommendation:

```bash
--stage1-v3-gain-update-mode adamw_projected --stage1-v3-gain-optimizer-weight 0.25
```

Saved contracts identify mode, coefficient and formula. Resume inherits
omitted values, ignores stale fresh-run environment defaults, and rejects
explicit mode/coefficient changes. Old contracts missing these fields keep
their legacy formula and original JSON bytes. Changing the formula requires
a fresh run; this does not authorize launching one.

## Worktree and launch

The local checkout is `/root/.codex/worktrees/541e/ILVR`; its branch is
`stage1-v3`. A native checkout supplies three explicit roots: local checkout,
remote container checkout and the corresponding remote host checkout. The
remote checkout must already exist with compatible reviewed code and venv.
The coordinator does not provision or silently substitute another checkout.

Example **dry run** (replace the remote roots with the provisioned checkout):

```bash
./run_multinode_stage1_training.sh \
  --run-id stage1-v3-example --zero-stage 1 --main-process-port 29689 \
  --local-repo-root /root/.codex/worktrees/541e/ILVR \
  --remote-repo-root /root/workspace/ILVR/.worktrees/stage1-v3 \
  --remote-host-repo-root /home/korea/Workspace/hjh/workspace/ILVR/.worktrees/stage1-v3 \
  --stage1-variant v3 --method dora \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --data-path /root/.codex/worktrees/541e/ILVR/data/TRAIN.jsonl \
  --image-root /root/.codex/worktrees/541e/ILVR/data/images_comt --dry-run
```

Remote machine rank 0 remains `korea@10.1.102.41`, container `llm_kd`, with
rendezvous `10.1.102.41` and Gloo/NCCL interface `bond1.102`; local machine rank
is 1. Two GPUs on each node are mandatory. Read the canonical AGENTS.md,
training-defaults.md and [operator guide](multinode-training.md) before an
actual run. Preserve the repository `data/`, `checkpoints/`, `venv/` links.
Results belong in `results/<run-id>/`, logs in `logs/`, all model state in
`checkpoints/<run-id>/`.

## Schedule and resume

One epoch uses the actual prepared-loader length, with ceil(microsteps/4)
updates. The final partial accumulation keeps the configured global-16
ordinary loss denominator. Gain instead divides by its actual global B.
The implementation does not import Stage 2 padding semantics or historical
431/212 update budgets.

`stage1_v3.json` records configuration/formula IDs (including the text selector), resolved profile and package
versions, frozen teacher content identity, retained source identity, sampler
settings, planned updates and remainder. Existing optimizer, scheduler, RNG
and EMA files remain authoritative. Only a complete nonfailed boundary may
be saved. Resume inherits the saved variant/profile when omitted; explicit
conflicts and historical standard→v3 reinterpretation are rejected. An omitted
selector inherits the saved mode. Older v3 contracts without a selector field
retain KL semantics; their files are not rewritten. Explicit KL↔entropy
switches or inconsistent selector/formula metadata fail before training;
changing modes requires a fresh run. Both
nodes receive identical contract bytes through the existing checkpoint
collection/redistribution route. Use a fresh coordinator `--attempt-id` when
resuming and preserve the run ID.

The contract also records `gain_weighting`, `gain_entropy_mode` and the Gain
formula identity. Omitted
resume weighting inherits the saved mode. Older contracts without this field
retain `none` and their original bytes; explicit `none`↔`token_entropy`
switches or mismatched formula IDs are rejected before model execution.
Contracts without `gain_entropy_mode` retain historical normalized entropy.
Saved alignment, Gain mode/coefficient (including zero), KL beta and warmup
remain authoritative. Fresh defaults never rewrite an existing checkpoint.
`Stage1V3Config` constructor omissions intentionally keep legacy decoding
semantics; fresh runtime entrypoints use `fresh_v3_config` and pass the fully
resolved configuration to the trainer.

## Bounded runtime acceptance

After separate execution authorization and both-node preflight, the fixed
route below runs tiny real Qwen/DoRA/BF16/ZeRO-1 via the owned supervisor:

```bash
./run_multinode_stage1_training.sh \
  --stage1-v3-runtime-acceptance --run-id stage1-v3-acceptance-example \
  --zero-stage 1 --main-process-port 29689 \
  --local-repo-root /root/.codex/worktrees/541e/ILVR \
  --remote-repo-root /root/workspace/ILVR/.worktrees/stage1-v3 \
  --remote-host-repo-root /home/korea/Workspace/hjh/workspace/ILVR/.worktrees/stage1-v3 \
  --dry-run
```

The two owned attempts perform an uninterrupted two-update baseline and an
interrupted one-update run followed by one resumed update. Every rank has six
microsteps with a 4+2 tail, and rank 0 has zero valid auxiliary blocks. Four
successful complete updates are compared. Afterwards, an injected overflow
decision exercises the real engine skip path. Separate injected nonfinite
and post-Adam Gain-write failures must prevent checkpoint publication; the
post-Adam failure is an additional failed attempt, not a fifth successful
complete update. These injections are not evidence of naturally occurring
overflow or failure.

The witness includes actual forward/VJP/site events, occurrence order,
master/BF16/moment checks and exact deterministic optimizer/scheduler/RNG/
teacher state comparisons across save/resume. Master comparisons use
atol 1e-6/rtol 1e-4; unchanged moments and BF16 casts require exact equality.
Tensor snapshots remain under checkpoints; JSON witnesses go under results.
`tests/stage1/v3_runtime_acceptance.py::validate_v3_rank_records` rejects
missing or inconsistent evidence. A successful dry run is only command
construction; full training and TEST400 remain separate follow-up work.

The acceptance builder prefixes each node's PATH with that node's
`venv/ilvr-venv/bin`, followed by the fixed system search paths. This applies
to initial and resumed attempts, including PATH-based child executable lookup.
The earlier remote PATH review finding is addressed; follow-up evidence is
recorded in `logs/stage1-v3-path-fix/workflow.md`.

## Stop after a complete checkpoint

For a fresh Stage1-v3 run, add `--stage1-v3-stop-after-checkpoint N` and
`--no-final-save` to the existing multi-node coordinator command. This optional
invocation setting defaults to off. It forces a full numbered checkpoint at
update N, then stops normally after Stage1 checkpoint saving, EMA/v3 sidecars
and rank synchronization finish. A save error remains a failed attempt.

The original one-epoch budget and LR schedule stay intact. N must be a positive
integer strictly below the budget derived from the actual prepared dataloader;
that comparison happens before the first update. Resume and model-only saves
cannot use this option. The Python spelling is
`--stage1_v3_stop_after_checkpoint N --no_final_save`.

Successful intentional stops emit schema-3 Stage1 producer evidence containing
`planned_stop: {global_step: N, max_steps: <full budget>, budget_complete: false}`.
The coordinator binds N to its invocation, collects all-rank checkpoint state,
and checks the full budget against both `trainer_state.json` and
`stage1_v3.json` before publishing the incomplete-budget training witness.
Ordinary producer evidence retains schema 2; historical schemas 1 and 2 remain
readable. The stop option is absent from the saved v3 objective/profile identity.

Resume with the same run ID, a new `--attempt-id`, and
`--resume <run-output>/checkpoint-N`; omit the stop option. The remaining updates
use the original saved budget. Checkpoint N remains available until resumption,
but a later checkpoint may rotate it away under the existing last-one policy.

CPU tests cover the actual save/callback/resume path and failure handling. They
do not establish four-rank ZeRO or Gain numerical equivalence. The subsequent
short runtime comparison should use 48 uniquely indexed presentations of the
longest sample (global batch 16, three updates): continuous 0→3 versus 0→2,
save/stop, resume 2→3. The first LR is zero, so the split must follow the second
update. Record nonzero applied FP32 displacement, Gain/projection execution,
optimizer/scheduler/RNG/teacher restoration and occurrence order. The older
acceptance harness above tests its original post-Adam/KL configuration and does
not replace that check for the optional optimizer Gain mode. Full training is
still a separate, unlaunched step.
