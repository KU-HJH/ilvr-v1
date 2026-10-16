# Stage1-v5 four-rank worker comparison

Understood as: extend the user's measured worker sweep through 64 to actual
Stage1-v5 training on the required two nodes/four GPU ranks, and determine
whether input-pipeline gains improve training throughput under rank competition.
User request: `64까지 4-rank로 확인해보자`.

Intake revision: a1c3d8557d08d2c1807b0a4800275e77b407c5e2 in the existing
host-managed `/root/.codex/worktrees/6b9d/ILVR` worktree, branch Stage1-v5.
No tracked changes at intake. The user's selected implementation method remains
subagent implementation with root coordination/verification and independent review.

Active phase: Superpowers bounded brainstorming, final design approval pending.
Readchk scope resolved from the preceding CPU-only comparisons; using-superpowers
and brainstorming applied. Canonical AGENTS.md, training-defaults.md and the
multi-node operator guide were read. This is a new development gate because
the existing CLI accepts workers but composition.py hardcodes Trainer workers=0.
The earlier throughput design explicitly excluded worker/profile changes.

Preliminary read-only resource check: both hosts respond, each has two idle
H100 NVL GPUs, 128 available CPUs and approximately 478–480 GiB available RAM.
This is not a launch preflight or four-rank participation witness.

## Frozen design proposal v1 — not approved

1. Make the existing Stage1 `--dataloader_num_workers` argument reach the actual
   Trainer configuration, retain default 0, and reject negative values. Preserve
   existing prefetch/persistence/pin-memory defaults, sampler and all objectives.
   Scope: composition.py, narrowly necessary argument validation, focused tests.
   Reuse the existing coordinator/launcher forwarding; no alternative launcher,
   profiler subsystem, new orchestration framework or automatic default adoption.
2. After implementation, independent review and required fresh verification,
   synchronize the reviewed revision into a clean compatible remote worktree and
   use the existing two-node/four-rank coordinator. Explicitly set the established
   Stage1-v5 recipe: original Qwen base, DoRA r32/alpha64/dropout0.05, ZeRO1,
   microbatch1/GAS4/global16, LR5e-5/seed42/warmup10/BF16, EMA1, CE0.1/alignment0.1,
   dil-bce-orig beta1/margin0/preference1, existing top-k and text-KL settings,
   latent8, SDPA decoder/FA2 vision and existing gradient checkpointing.
3. Sweep workers per rank 0,2,4,8,16,32,64. Each cell starts fresh from the same
   base/seed and full real TRAIN data, runs 20 optimizer updates (320 sample
   presentations), and preserves the same sample order. Treat the first 5 updates
   as timing warmup; compare aggregate elapsed time for updates6–20 using existing
   progress logs, total Trainer time separately, and host/GPU resource telemetry.
   Existing final checkpoint behavior stays intact; exclude save time from the
   progress-window metric. Keep model state only under checkpoints, logs under
   logs, metadata under results, with unique run IDs. Do not evaluate TEST400.
4. Capture resolved values on both nodes and runtime participation of ranks0–3.
   Verify the requested worker count reaches prepared loaders/runtime contracts,
   and check sample/order equivalence at the relevant loader boundary with focused
   tests. A CLI string alone is not worker execution evidence. Report timing
   granularity and single-run variability; if cells are too close to rank, report
   inconclusive rather than claiming an optimum. No full-training performance
   claim follows from the earlier CPU-only comparisons.
5. Read existing runtime evidence before adding measurement code. The current
   proposal uses existing progress/Trainer timing, which does not attribute exact
   input-wait duration. The primary answer is wall-clock training throughput under
   four-rank competition. Production worker defaults remain 0 unless separately
   selected by the user.

Remaining uncertainties: high-worker CPU gains may be hidden by GPU compute;
short windows and integer-second progress timestamps can obscure small gains;
forked workers may behave differently after full CUDA/model initialization.
The sweep itself is requested, but its necessary code change and this concrete
design are not yet finally approved. No implementation or training has started.

## Root hate pass on frozen proposal v1

```json
{
  "root": "A short four-rank wall-time sweep can mistake changed sample/RNG execution, startup, or coarse logging noise for a worker throughput effect; then even a complete sweep would not answer whether the same training work became faster.",
  "first_nail": "Before interpreting the sweep, use the existing real prepared-loader boundary to compare the requested worker count, first-window sample identities/order and initial objective evidence for the 0 and 64 cells. Compare existing progress timestamp resolution with the observed difference. If the work differs or the timing uncertainty spans the claimed gain, reject the ranking rather than expanding or declaring an optimum."
}
```

First nail is a proposed falsification, not evidence or authority to launch.
The independent reads below assess this frozen version; root will synthesize
their findings into a separate design commitment before final human approval.

## Macrothink independent reads and root classification

Root invoked three same-model fresh contexts with the underlying goal, fixed
topology/profile, current hardcoded worker fact, and a request to inspect runtime
before reading the frozen proposal. All were read-only, with no skills, nested
agents, implementation or training permitted.

- `/root/workers4r_fresh_b`: divergent-compatible. At 20 updates, each rank consumes
  80 microbatches (60 measured), while workers64/prefetch2 can schedule 128. A
  prepared backlog may cover the measured window; this cannot establish sustained
  input throughput or a general optimum. Also inspect the loader actually iterated,
  its nested Accelerate sampler, and observed order; the v5 contract separately
  constructs a loader before Trainer constructs the real one. CPU-only collation
  and native seeding do not prove CUDA-after-fork safety without runtime evidence.
- `/root/workers4r_fresh_a`: convergent, with compatible precision limits. Derive
  `(first elapsed at 20/20 - elapsed at 5/20) / 15`; do not use smoothed s/it or a
  later final redraw that includes checkpoint time. Trainer duration includes its
  step checkpoint even though a separate final save is timed outside it.
- `/root/workers4r_fresh_c`: convergent, with compatible resource/timing limits.
  Workers64 means 128 loader processes per node and 256 overall. Check both hosts'
  RAM and shared memory. Preserve normal checkpoint evidence required by the
  coordinator. A step-begin/end-only callback would omit prior loader fetches.

No divergent-incompatible direction was identified for the explicitly bounded
short-run comparison. The common material finding is the finite prefetch backlog:
convergence is reassurance only and does not establish correctness or speed.

## Design commitment v1 — pending final human approval

Agreed problem: compare actual Stage1-v5 training under four-rank CPU/GPU resource
competition for workers0/2/4/8/16/32/64. Preserve the implementation method already
selected by the user: subagent implementer, independent reviewer, root validation.

Final design is proposal v1 with the following explicit measurement refinements:

- Use native positive-worker defaults (prefetch2, current persistence/pinning),
  and treat worker count as per rank. Record actual prepared-loader behavior,
  ordered sample evidence and rank participation; do not infer them from CLI.
- Each cell runs20 updates, with5 timing warmup and15 measured, from identical
  initial base/seed and data. Compare the first completed20/20 cumulative elapsed
  record against5/20, retaining raw carriage-return logs. Report Trainer time
  separately as inclusive of its normal checkpoint. No custom profiler is added.
- Limit conclusions to this short-run comparison. Sixty measured microbatches per
  rank do not exceed the possible128-batch prefetch backlog at workers64. Small
  differences within logging resolution or variability remain inconclusive; no
  production optimum or full-epoch speedup is claimed. The user is asked to accept
  this disclosed scope when approving, rather than silently treating it as steady
  state evidence.
- No underlying performance advantage is presumed. First-nail evidence remains
  pending: check actual worker execution, order/initial objective and usable timing
  boundaries at execution, before interpreting or publishing a ranking. CPU
  tests must cover argument propagation and relevant sample/RNG behavior. Runtime
  failures remain failures; no fallback to another topology/profile.

Pre-approval operational evidence: existing coordinator dry-run succeeded for
workers64/max_steps20/world4/ZeRO1, logged at
`logs/stage1-v5-workers4r-20261016/proposal-v1-dry-run.log`. This confirms forwarding
only, not that the hardcoded Trainer value is fixed. Both nodes' TRAIN.jsonl hash
matches `5bc1313a594f5713975b67388c745c6632275c37fd260055b27230c9cfedbed8`.
Remote Stage1-v5 worktree is clean at148b0e8, so the reviewed descendant must be
synchronized before launch. Both repository Python versions are3.11.15+; package
and execution-code compatibility still require final preflight. Checkpoint
storage is ample (local37TiB, remote39TiB). No product changes or training occurred.

Final approval gate: the previous throughput commitment explicitly preserved
workers0 and excluded worker/profile changes. The current request authorizes the
experiment, but the mandatory bounded workflow still requires approval of this
new concrete design before implementation. Applicable skill instruction:
`brainstorming/SKILL.md`: "Bounded: the human partner approves the short in-chat
design." Final approval had not been received at proposal presentation.

## Approval and implementation handoff

User replied `ㅇㅋ` to the exact final in-chat design v1 (worker forwarding;
0/2/4/8/16/32/64 per rank;20 updates,5 excluded; short-window scope). Approval
binds that design and its disclosed steady-state limitation. Brainstorming is
complete; active phase is bounded subagent-driven implementation, then independent
task/final reviews, fresh verification and the authorized experiment.

Writing-plans is nonapplicable: this is one bounded existing-path correction,
recorded in the concrete task brief. Existing linked worktree verified via distinct
git-dir/common-dir, no superproject, named Stage1-v5 branch. Environment already
provisioned; no dependency installation. Existing full-suite baseline on identical
product content:4642 passed/11 skipped/1 warning, recorded in the prior throughput
workflow and root-full-suite.log. No repeated broad baseline run; root owns one
fresh full-suite completion verification after review, per standing user policy.
