# Stage1-v3

`stage1-v3` adds the text-only restriction objective from
`documents/text_only_gain_v1.md` to the existing Stage 1 path. Select it with
`--stage1-variant v3` on the coordinator/launcher, or `--stage1_variant v3`
on the training CLI. An omitted variant on a fresh run retains `standard`.

The implementation is undergoing CPU verification and independent review.
**Distributed acceptance has not been run.** Tiny CPU checks do not establish
four-rank execution, full 3B memory use, throughput, or TEST400 accuracy.

## Profile and objective

Fresh v3 uses the original Qwen2.5-VL-3B-Instruct base, DoRA r32/alpha64/dropout
0.05 with q/k/v/o/gate/up/down decoder projections, latent size 8, CE weight
0.1, alignment weight 1, frozen EMA 1.0 and helper noise 0.01. Execution uses
two nodes/four ranks, microbatch 1, accumulation 4, global batch 16, LR 5e-5,
seed 42, warmup 10, BF16, AdamW/linear, weight decay 0.01, clipping 1.0,
ZeRO-1 without offload, SDPA and the existing Stage 1 vision path,
non-reentrant checkpointing, max sequence 32768 and zero data workers.
The full profile is explicitly resolved after sourced environment settings.

Each valid immediate text block has independent R and RV replay branches.
R excludes selected prior text keys at scored queries; RV additionally blocks
the current latent body for the suffix. START/END and the original image stay
available. Source-item crossing tokens are excluded only from auxiliary spans.
The collator's input tensors and CE labels retain their existing meaning.

Text selection uses FP32 full-vocabulary student-to-reference KL at temperature
2, centered within each sample, rho 0.3–0.5, text cap 0.3, fallback 0.1 and
image cap 0. The ordinary loss adds temperature-2 KL(detached F || R), averaged
by query within each valid sample then across samples. Its weight rises to
0.01 over ceil(0.05 × planned updates).

R and RV GT scores are block means. Their gradient sums and block count B are
reduced across all four ranks. With g=(R−RV)/B and h=RV/B, a negative global
h·g component is projected away using FP64 dot/norm reductions. Tiny nonzero
h norm squared below 1e-30 produces no displacement. After a successful AdamW
step, alpha=0.001 times the direction is added to FP32 master parameters and
synchronized to BF16; Adam moments remain unchanged by this addition.
An engine-skipped step clears the auxiliary window without applying Gain.

Core execution is F once, gain-clean once, frozen reference once and one packed
R/RV decoder call. Helper work and checkpoint recomputation are counted
separately. Joint two-lane R/KL VJP and a separate RV VJP preserve attached
prefix gradients and original sample/token dropout identity. No valid blocks
means ordinary F/helper execution with zero auxiliary work.

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

`stage1_v3.json` records configuration/formula IDs, resolved profile and package
versions, frozen teacher content identity, retained source identity, sampler
settings, planned updates and remainder. Existing optimizer, scheduler, RNG
and EMA files remain authoritative. Only a complete nonfailed boundary may
be saved. Resume inherits the saved variant/profile when omitted; explicit
conflicts and historical standard→v3 reinterpretation are rejected. Both
nodes receive identical contract bytes through the existing checkpoint
collection/redistribution route. Use a fresh coordinator `--attempt-id` when
resuming and preserve the run ID.

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
